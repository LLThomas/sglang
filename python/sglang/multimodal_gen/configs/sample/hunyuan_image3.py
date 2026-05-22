import json
import os
from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.sample.sampling_params import DataType, SamplingParams
from sglang.multimodal_gen.configs.sample.teacache import TeaCacheParams


@dataclass
class HunyuanImage3SamplingParams(SamplingParams):
    num_inference_steps: int = 50
    height: int = 1024
    width: int = 1024
    guidance_scale: float = 2.5
    seed: int = 42
    cot_text: str = ""  # DiT-only 模式：由调用者提供 CoT 文本
    data_type: DataType = DataType.IMAGE

    # AR stage params
    bot_task: str = ""  # "", "image", "think", "recaption", "think_recaption"
    sequence_template: str = "pretrain"  # "pretrain" or "instruct"
    system_prompt: str = ""  # system prompt (auto-selected by bot_task if empty)
    ar_max_new_tokens: int = 2048  # max tokens for AR generation
    ar_temperature: float = 0.6  # AR sampling temperature (0.0 = greedy)
    ar_top_p: float = 0.95  # AR top-p
    ar_top_k: int = 1024  # AR top-k
    ar_progress_log_interval: int = 16  # 0 disables AR token progress logs
    drop_think: bool = False  # drop think portion from CoT
    image_size: str = "auto"  # "auto" = predict ratio via AR, or "HxW"
    sys_type: str = "dynamic"  # system prompt type: "None", "en_vanilla", "en_recaption", "en_think_recaption", "en_unified", "dynamic", "custom"

    # TeaCache params for denoising acceleration.
    # Coefficients calibrated via polyfit on 3920 data points (80 prompts × 49 steps)
    # from vllm-omni's HunyuanImage3 TeaCache config.
    # Set ``enable_teacache=True`` via the API or JSON config to activate.
    teacache_params: TeaCacheParams = field(
        default_factory=lambda: TeaCacheParams(
            teacache_thresh=0.2,
            coefficients=[
                1.04117826e02,
                -1.26848482e02,
                5.68168652e01,
                -1.04182570e01,
                6.78098549e-01,
            ],
        )
    )

    def _adjust(self, server_args):
        self._apply_generation_config_defaults(server_args)
        super()._adjust(server_args)

    def _apply_generation_config_defaults(self, server_args):
        """Apply official HunyuanImage3 generation defaults when not overridden."""
        explicit_fields = getattr(self, "_explicit_fields", set()) or set()
        if not getattr(server_args, "model_path", None):
            return
        config_path = os.path.join(server_args.model_path, "generation_config.json")
        if not os.path.exists(config_path):
            return

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                generation_config = json.load(f)
        except Exception:
            return

        field_map = {
            "diff_infer_steps": "num_inference_steps",
            "diff_guidance_scale": "guidance_scale",
            "sequence_template": "sequence_template",
            "bot_task": "bot_task",
            "drop_think": "drop_think",
            "max_new_tokens": "ar_max_new_tokens",
            "temperature": "ar_temperature",
            "top_p": "ar_top_p",
            "top_k": "ar_top_k",
        }
        for config_key, field_name in field_map.items():
            if (
                field_name not in explicit_fields
                and config_key in generation_config
                and generation_config[config_key] is not None
            ):
                setattr(self, field_name, generation_config[config_key])

        if "sys_type" not in explicit_fields:
            use_system_prompt = generation_config.get("use_system_prompt")
            if use_system_prompt is not None:
                self.sys_type = str(use_system_prompt)
