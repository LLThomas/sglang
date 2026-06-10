# HY3 分阶段实现路线图

> 参考 vllm-omni 三阶段路线图，将 HY3 支持拆分为独立 PR 分阶段提交。

---

## 总览

| Phase | 内容 | 状态 | 关键文件 |
|-------|------|------|---------|
| **1** | DiT-Only T2I 基线 | 🔄 进行中 | DiT model, Stage, Pipeline |
| **2** | AR 生成 + cfg_rank broadcast | ⬜ 待开始 | AR tokenizer, Stage AR methods |
| **3** | CFG-Parallel 去噪 | ⬜ 待开始 | Stage forward cfg_rank 分支 |
| **4** | Stage 拆分重构（可选） | ⬜ 待开始 | 新建多个 sub-stage 文件 |
| **5** | 性能优化 | ⬜ 待开始 | DiT model |

```
Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5
```

---

## Phase 1: DiT-Only T2I 基线

**目标**：单卡 + TP-only 的 T2I 正确工作，无 AR，无 CFG-Parallel。

**分支策略**：从 `origin/main` 创建 `hyimage3/phase1`，手动添加精简版文件。

### 文件清单

| 文件 | 来源 | 处理 |
|------|------|------|
| `configs/models/dits/hunyuan_image3.py` | 直接复制 | 完整文件 |
| `configs/models/vaes/hunyuan_image3.py` | 直接复制 | 完整文件 |
| `configs/pipeline_configs/hunyuan_image3.py` | 精简 | 去掉 AR 相关配置 |
| `configs/sample/hunyuan_image3.py` | 精简 | 只保留 `guidance_scale`, `num_inference_steps`, `seed`, `image_size`, `height`, `width` |
| `runtime/models/dits/hunyuan_image3.py` | 精简 | 去掉 `generate_text()`, `_forward_ar_text()`, `prepare_ar_input_ids()` |
| `runtime/models/vaes/autoencoder_kl_hunyuanimage3.py` | 直接复制 | 完整文件 |
| `runtime/models/encoders/siglip2.py` | 直接复制 | 完整文件 |
| `runtime/pipelines/hunyuan_image3_pipeline.py` | 精简 | 去掉 AR 相关逻辑 |
| `runtime/pipelines_core/stages/model_specific_stages/hunyuan_image3.py` | 精简 | 去掉所有 AR 方法 |
| `runtime/pipelines_core/stages/model_specific_stages/hunyuan_image3_tokenizer.py` | **不包含** | Phase 2 |
| `runtime/pipelines_core/stages/denoising.py` | 直接复制 | 完整文件 |
| `runtime/distributed/srt_moe_bridge.py` | 直接复制 | 完整文件 |
| `runtime/distributed/parallel_state.py` | 小改动 | 只保留基础改动 |
| `runtime/managers/gpu_worker.py` | 小改动 | 完整文件 |
| `runtime/server_args.py` | 精简 | 去掉 AR 相关参数 |
| `runtime/cache/cache_dit_integration.py` | 直接复制 | 完整文件 |
| `runtime/cache/teacache.py` | 小改动 | 完整文件 |
| `registry.py` | 直接复制 | 完整文件 |
| `python/sglang/utils.py` | 小改动 | 完整文件 |

### 需要删除的 AR 相关代码

**DiT model** (`hunyuan_image3.py`):
- `generate_text()` 方法 (~500 行)
- `_forward_ar_text()` 方法
- `prepare_ar_input_ids()` 方法
- `get_ar_model()` 等相关辅助

**Stage** (`hunyuan_image3.py`):
- `_run_ar_generation()` 方法
- `_run_ratio_prediction()` 方法
- `_parse_ar_output()` 方法
- `_build_ar_logits_processor()` 方法
- `_cot_stop_token_ids()` 方法
- `_build_stage_transitions()` 方法
- `forward()` 中对 AR 的调用

**Tokenizer** (`hunyuan_image3_tokenizer.py`):
- **整个文件不包含在 Phase 1**

**Sampling params** (`configs/sample/hunyuan_image3.py`):
- 去掉 `bot_task`, `ar_max_new_tokens`, `ar_temperature`, `ar_top_p`, `ar_top_k`, `drop_think`, `system_prompt`

### 测试

- 单卡 `--num-gpus 1 --image-size 1024x1024`（不设 bot_task）
- TP=2/TP=4 同上，验证输出一致

---

## Phase 2: AR 生成 + cfg_rank broadcast

**目标**：添加 AR 支持，并修复 TP+CFG 场景下 AR 输出不一致问题。

**分支策略**：从 `origin/main` 创建 `hyimage3/phase2`，在 Phase 1 基础上添加 AR 代码。

### 新增文件

| 文件 | 行数 | 说明 |
|------|------|------|
| `runtime/pipelines_core/stages/model_specific_stages/hunyuan_image3_tokenizer.py` | 1008 | AR tokenizer |

### 追加代码

**DiT model**:
- `generate_text()` 方法
- `_forward_ar_text()` 方法
- `prepare_ar_input_ids()` 方法

**Stage**:
- `_run_ar_generation()` 方法
- `_run_ratio_prediction()` 方法
- `_parse_ar_output()` 方法
- `_build_ar_logits_processor()` 方法
- `_cot_stop_token_ids()` 属性
- `_build_stage_transitions()` 方法
- `forward()` 中 cfg_rank 条件执行 + broadcast

**Sampling params**:
- 添加 `bot_task`, `ar_max_new_tokens`, `ar_temperature`, `ar_top_p`, `ar_top_k`, `drop_think`, `system_prompt`

### cfg_rank broadcast 模式

```python
# 在 _run_ar_generation() 中
cfg_rank = get_classifier_free_guidance_rank()
cfg_group = get_cfg_group()

if cfg_rank == 0:
    # 执行 AR 生成
    ar_result = [cot_text, ratio_index]
else:
    ar_result = [None, None]

# Broadcast 结果
if cfg_group is not None:
    ar_result = broadcast_pyobj(
        ar_result, cfg_group.rank,
        cfg_group.cpu_group, src=cfg_group.ranks[0],
    )

batch.cot_text = ar_result[0]
if ar_result[1] is not None:
    batch.ratio_index = ar_result[1]
```

### 测试

- 8卡 TP=4+CFG=2 + `--bot-task think`：两 TP group CoT 一致
- 单卡/TP-only + bot_task：行为不变

---

## Phase 3: CFG-Parallel 去噪

**目标**：每个 cfg_rank 只构建对应分支，cond/uncond 并行执行，吞吐量 ~2x。

### 改动

**Stage** — `hunyuan_image3.py` (Stage) 的 `forward()`:
- `cfg_rank=0`：只嵌入 cond embedding/RoPE/mask
- `cfg_rank=1`：只嵌入 uncond embedding/RoPE/mask

**Pipeline Config**:
- 确保 `prepare_neg_cond_kwargs` 处理 None 分支

### 测试

- TP=4+CFG=2 + `--enable-cfg-parallel`：吞吐量 ~2x，输出一致
- 单卡/TP-only：不受影响

---

## Phase 4: Stage 拆分重构（可选）

**目标**：纯重构，将 2200 行 BeforeDenoising 拆为可维护的子 stage。

| 子 Stage | 职责 |
|----------|------|
| `HunyuanImage3ARStage` | AR 生成 + ratio 预测 |
| `HunyuanImage3PreprocessStage` | 输入解析、token 序列构建 |
| `HunyuanImage3EmbedStage` | Token embedding、RoPE、mask |
| `HunyuanImage3NoiseInitStage` | 噪声初始化、scheduler |

---

## Phase 5: 性能优化（后续）

1. **Packed Attention**：多请求 batch 消除 padding
2. **EP (Expert Parallelism)**：MoE 层专家并行
3. **TeaCache 调优**：优化 skip 阈值
4. **AR 性能**：AR 阶段 transformer 驻留优化

---

## 实施进度

### Phase 1 进度

- [x] 创建 `hyimage3-phase1` 分支（从 `origin/main` 创建）
- [x] 添加核心配置文件（DiT config, VAE config, Pipeline config）
- [x] 添加 DiT model（精简版：去掉 `generate_text`, `_forward_ar_text`, `_compute_ar_logits`）
- [x] 添加 VAE + siglip2 encoder
- [x] 添加 Stage（精简版：去掉所有 AR 方法，1359 行 vs 原始 2199 行）
- [x] 添加 Pipeline + Sample config（精简版：去掉 AR 参数）
- [x] 添加 Server args + parallel_state + MoE bridge + cache 等小改动
- [x] 所有文件语法检查通过
- [ ] 测试单卡 T2I
- [ ] 测试 TP T2I
- [ ] 提交 PR

**统计数据**：18 文件，5109 行插入（vs hyimage3 全量 19 文件 7268 行）