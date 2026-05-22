from dataclasses import dataclass, field

import torch

from sglang.multimodal_gen.configs.models import DiTConfig, VAEConfig
from sglang.multimodal_gen.configs.models.dits.hunyuan_image3 import HunyuanImage3DiTConfig
from sglang.multimodal_gen.configs.models.vaes.hunyuan_image3 import HunyuanImage3VAEConfig
from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ImagePipelineConfig,
    ModelTaskType,
)


@dataclass
class HunyuanImage3PipelineConfig(ImagePipelineConfig):
    """Pipeline config for HunyuanImage-3.0 (Hybrid style).

    AR generation is controlled at runtime by ``batch.bot_task`` and
    consolidated into BeforeDenoisingStage, not via a pipeline-level flag.
    """

    # HunyuanImage-3.0 supports both text-only and text+image generation.  In
    # sglang, TI2I means image input is accepted but not required.
    task_type: ModelTaskType = ModelTaskType.TI2I
    vae_precision: str = "fp16"
    should_use_guidance: bool = True
    vae_tiling: bool = False
    enable_autocast: bool = False
    vae_sp: bool = False
    flow_shift: float = 3.0
    tie_word_embeddings: bool = False  # official default is False

    dit_config: DiTConfig = field(default_factory=HunyuanImage3DiTConfig)
    vae_config: VAEConfig = field(default_factory=HunyuanImage3VAEConfig)

    def get_freqs_cis(self, batch, device, rotary_emb, dtype):
        """返回 batch 中预计算的 2D RoPE。"""
        return batch.rope_2d

    def get_pos_prompt_embeds(self, batch):
        """覆写：返回完整 token 序列嵌入（不仅是文本）。

        在 HunyuanImage3 的单流架构中，encoder_hidden_states
        包含完整 token 序列（文本 + 图像占位符）。
        DenoisingStage 的合并顺序为：
          base | prepare_pos_cond_kwargs() | {encoder_hidden_states: get_pos_prompt_embeds()}
        因此 encoder_hidden_states 必须通过此方法设置。
        """
        return batch.prompt_embeds[0]

    def get_neg_prompt_embeds(self, batch):
        """覆写：返回无条件 token 序列嵌入。"""
        return batch.negative_prompt_embeds[0]

    def prepare_pos_cond_kwargs(self, batch, device, rotary_emb, dtype):
        # 注意：encoder_hidden_states 由 get_pos_prompt_embeds() 处理
        # 不要在此设置，因为 DenoisingStage 合并时会被覆盖
        kwargs = {
            "encoder_attention_mask": batch.encoder_attention_mask,
            "image_mask": batch.image_mask,
            "image_scatter_index": batch.image_scatter_index,
            "timestep_scatter_index": batch.gen_timestep_scatter_index
            if batch.gen_timestep_scatter_index is not None
            else batch.timestep_scatter_index,
            "cond_vae_image_mask": batch.cond_vae_image_mask,
            "cond_vae_scatter_index": batch.cond_vae_scatter_index,
            "cond_vit_image_mask": batch.cond_vit_image_mask,
            "cond_vit_scatter_index": batch.cond_vit_scatter_index,
            "cond_vit_embeds": batch.cond_vit_embeds,
            "rope_2d": batch.rope_2d,
            "position_ids": batch.position_ids,
            "token_h": batch.token_h,
            "token_w": batch.token_w,
        }
        # Pass cond_vae_images and cond_timestep for TI2I
        if hasattr(batch, "cond_vae_images") and batch.cond_vae_images is not None:
            kwargs["cond_vae_images"] = batch.cond_vae_images
        if hasattr(batch, "cond_timestep") and batch.cond_timestep is not None:
            kwargs["cond_timestep"] = batch.cond_timestep
        if (
            hasattr(batch, "cond_timestep_scatter_index")
            and batch.cond_timestep_scatter_index is not None
        ):
            kwargs["cond_timestep_scatter_index"] = batch.cond_timestep_scatter_index
        return kwargs

    def prepare_neg_cond_kwargs(self, batch, device, rotary_emb, dtype):
        # 注意：encoder_hidden_states 由 get_neg_prompt_embeds() 处理
        # ti2i CFG: the source image is a FIXED condition present in BOTH
        # branches (only text differs), matching vllm-omni which repeats
        # cond images cfg_factor times. So scatter the SAME cond features at
        # the uncond branch's own positions (neg_cond_* masks/indices).
        kwargs = {
            "encoder_attention_mask": batch.neg_encoder_attention_mask,
            "image_mask": batch.neg_image_mask,
            "image_scatter_index": batch.neg_image_scatter_index,
            "timestep_scatter_index": batch.neg_gen_timestep_scatter_index
            if batch.neg_gen_timestep_scatter_index is not None
            else batch.neg_timestep_scatter_index,
            "cond_vae_image_mask": getattr(batch, "neg_cond_vae_image_mask", None),
            "cond_vae_scatter_index": getattr(batch, "neg_cond_vae_scatter_index", None),
            "cond_vit_image_mask": getattr(batch, "neg_cond_vit_image_mask", None),
            "cond_vit_scatter_index": getattr(batch, "neg_cond_vit_scatter_index", None),
            "cond_vit_embeds": getattr(batch, "cond_vit_embeds", None),
            "cond_timestep_scatter_index": getattr(batch, "neg_cond_timestep_scatter_index", None),
            "rope_2d": batch.neg_rope_2d,
            "position_ids": batch.neg_position_ids,
            "token_h": batch.token_h,
            "token_w": batch.token_w,
        }
        # Same source image features as the cond branch (ti2i CFG keeps image in both)
        if hasattr(batch, "cond_vae_images") and batch.cond_vae_images is not None:
            kwargs["cond_vae_images"] = batch.cond_vae_images
        if hasattr(batch, "cond_timestep") and batch.cond_timestep is not None:
            kwargs["cond_timestep"] = batch.cond_timestep
        return kwargs

    def get_decode_scale_and_shift(self, device, dtype, vae):
        """返回 VAE 解码的 scale 和 shift。"""
        scale = self.vae_config.arch_config.scaling_factor
        shift = getattr(self.vae_config.arch_config, "shift_factor", 0.0)
        if isinstance(scale, torch.Tensor):
            scale = scale.to(device=device, dtype=dtype)
        if isinstance(shift, torch.Tensor):
            shift = shift.to(device=device, dtype=dtype)
        return scale, shift

    def post_denoising_loop(self, latents, batch):
        return latents.to(torch.bfloat16)

    # -- Sequence Parallelism --------------------------------------------------
    # HY3 does not use Sequence Parallelism (SP handled by TP).  Latents remain
    # [B, C, H, W] on every rank, so the base-class sharding/gathering must be
    # overridden as no-ops.

    def shard_latents_for_sp(self, batch, latents):
        return latents, False

    def gather_latents_for_sp(self, latents, batch=None):
        return latents

    def preprocess_decoding(self, latents, server_args=None, vae=None):
        """Add temporal dimension for 3D VAE decode.

        HunyuanImage-3.0 uses a 3D VAE that expects 5D input (B, C, T, H, W).
        For single-frame image generation, we unsqueeze at dim=2 to add T=1.
        """
        if latents.ndim == 4:
            latents = latents.unsqueeze(2)  # (B, C, H, W) -> (B, C, 1, H, W)
        return latents

    @staticmethod
    def post_decoding(frames, server_args=None):
        """Remove temporal dimension after 3D VAE decode.

        After VAE decode, single-frame images have T=1 in dim=2.
        We squeeze it back to 4D (B, C, H, W) for output.
        """
        if frames.ndim == 5 and frames.shape[2] == 1:
            frames = frames.squeeze(2)  # (B, C, 1, H, W) -> (B, C, H, W)
        return frames

