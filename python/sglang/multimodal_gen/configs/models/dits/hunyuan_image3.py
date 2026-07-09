from dataclasses import dataclass, field

import torch

from sglang.multimodal_gen.configs.models.dits.base import DiTArchConfig, DiTConfig
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum


@dataclass
class HunyuanImage3ArchConfig(DiTArchConfig):
    _compile_conditions: list = field(default_factory=list)
    _supported_attention_backends: set[AttentionBackendEnum] = field(
        default_factory=lambda: {
            AttentionBackendEnum.FA,
            AttentionBackendEnum.TORCH_SDPA,
            AttentionBackendEnum.SAGE_ATTN,
        }
    )
    param_names_mapping: dict = field(
        default_factory=lambda: {
            # Strip leading "model." prefix used by HF transformers format
            r"^model\.(.*)$": r"\1",
            # HF uses "gate_and_up_proj"; sglang model uses "gate_up_proj"
            r"(.*)\.gate_and_up_proj\.(.*)": r"\1.gate_up_proj.\2",
            # HF uses "ln_f"; sglang model uses "norm"
            r"^ln_f\.": r"norm.",
            # HF uses "wte"; sglang model uses "embed_tokens"
            r"^wte\.": r"embed_tokens.",
            # HF uses "mlp.gate.wg"; sglang model uses "mlp.gate"
            r"(.*)\.mlp\.gate\.wg\.(.*)": r"\1.mlp.gate.\2",
        }
    )
    reverse_param_names_mapping: dict = field(
        default_factory=lambda: {
            # Add back "model." prefix for HF format
            r"^(.*)$": r"model.\1",
        }
    )

    # Architecture params (from HunyuanImage-3.0 config.json)
    hidden_size: int = 4096
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    attention_head_dim: int = 128
    intermediate_size: int = 3072
    patch_size: int = 1
    in_channels: int = 32  # VAE latent_channels from config.json
    out_channels: int = 32
    vocab_size: int = 133120
    image_base_size: int = 1024
    vae_downsample_factor: tuple[int, int] = (16, 16)
    rope_theta: float = 10000.0
    rope_axes_dim: tuple[int, int] = (64, 64)
    qk_norm: str = "rms_norm"
    guidance_embeds: bool = False
    text_embed_dim: int = 3584  # Qwen2.5-VL hidden size
    # GQA params
    num_key_value_heads: int = 8
    rms_norm_eps: float = 1e-5
    attention_bias: bool = False
    # MoE params (per-layer lists from config.json)
    num_experts: int = 64
    moe_topk: int | list[int] = field(default_factory=lambda: [8] * 32)
    num_shared_expert: int | list[int] = field(default_factory=lambda: [1] * 32)
    moe_layer_num_skipped: int = 0
    use_mixed_mlp_moe: bool = True
    moe_intermediate_size: int | list[int] = field(
        default_factory=lambda: [3072] * 32
    )
    # patch_embed / final_layer
    patch_embed_hidden_dim: int = 1024
    mlp_ratio: float = 4.0
    # Timestep embedders
    timestep_embed_dim: int = 4096
    # Special token IDs from config.json
    bos_token_id: int = 127958
    eos_token_id: int = 127957
    pad_token_id: int = 128009
    image_token_id: int = 128006
    # lm_head config
    tie_word_embeddings: bool = False

    def __post_init__(self):
        super().__post_init__()
        self.num_channels_latents = self.in_channels


@dataclass
class HunyuanImage3DiTConfig(DiTConfig):
    arch_config: DiTArchConfig = field(default_factory=HunyuanImage3ArchConfig)
    prefix: str = "hunyuanimage"
