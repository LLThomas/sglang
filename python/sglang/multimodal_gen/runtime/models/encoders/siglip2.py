# Adapted from vllm-omni:
# https://github.com/Tencent-Hunyuan/HunyuanImage-3.0/blob/main/vllm_omni/model_executor/models/hunyuan_image3/siglip2.py
# Ported to sglang's ImageEncoder base class for HunyuanImage-3.0.

# SPDX-License-Identifier: Apache-2.0
import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.multimodal_gen.configs.models.encoders import (
    BaseEncoderOutput,
    ImageEncoderConfig,
)
from sglang.multimodal_gen.runtime.layers.activation import get_act_fn
from sglang.multimodal_gen.runtime.loader.weight_utils import default_weight_loader
from sglang.multimodal_gen.runtime.models.encoders.base import ImageEncoder
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Config helper -- simple attribute holder from vllm-omni
# ---------------------------------------------------------------------------


class Config:
    """Dict-like config wrapper used by the internal SigLIP2 sub-modules."""

    def __init__(self, config: dict | None = None):
        if config is not None:
            for key, value in config.items():
                setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key, None)

    def __setitem__(self, key, value):
        return setattr(self, key, value)


# ---------------------------------------------------------------------------
# _prepare_4d_attention_mask  (replaces transformers.modeling_utils import)
# ---------------------------------------------------------------------------


def _prepare_4d_attention_mask(
    attention_mask: torch.Tensor,
    dtype: torch.dtype,
    target_len: int | None = None,
):
    """Create a 4D attention mask from a 2D mask.

    Args:
        attention_mask: [batch_size, seq_len] with 1 = valid, 0 = pad.
        dtype: Target dtype for the output mask.
        target_len: Optional target sequence length (for cross-attention).

    Returns:
        [batch_size, 1, target_len, seq_len] mask where 0 = attend,
        -inf = masked out.
    """
    batch_size, seq_len = attention_mask.shape
    if target_len is None:
        target_len = seq_len

    # Expand to [batch_size, 1, target_len, seq_len]
    expanded_mask = attention_mask[:, None, None, :].expand(
        batch_size, 1, target_len, seq_len
    )

    # Convert: 1 -> 0.0 (attend), 0 -> -inf (mask out)
    inverted_mask = 1.0 - expanded_mask.float()
    return inverted_mask.masked_fill(inverted_mask.bool(), float("-inf")).to(dtype)


# ---------------------------------------------------------------------------
# SigLIP2 Vision Embeddings
# ---------------------------------------------------------------------------


class Siglip2VisionEmbeddings(nn.Module):
    """Vision embeddings for SigLIP2 -- uses a linear patch projection instead
    of the Conv2d used by the original SigLIP / CLIP."""

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.num_channels = config.num_channels
        self.num_patches = config.num_patches

        self.patch_embedding = nn.Linear(
            self.num_channels * self.patch_size * self.patch_size,
            self.embed_dim,
        )

        self.position_embedding = nn.Embedding(self.num_patches, self.embed_dim)

    @staticmethod
    def resize_positional_embeddings(
        position_embedding: nn.Embedding,
        old_num_patches: int,
        new_num_patches: int,
    ) -> nn.Embedding:
        """Resize positional embeddings by interpolation."""
        if old_num_patches == new_num_patches:
            return position_embedding

        old_weight = position_embedding.weight.data
        embed_dim = old_weight.shape[-1]

        # Interpolate from old grid to new grid
        old_weight = old_weight.reshape(1, old_num_patches, embed_dim).permute(0, 2, 1)
        new_weight = F.interpolate(
            old_weight, size=new_num_patches, mode="nearest"
        ).permute(0, 2, 1)
        new_weight = new_weight.reshape(new_num_patches, embed_dim)

        new_embedding = nn.Embedding(new_num_patches, embed_dim)
        new_embedding.weight.data.copy_(new_weight)
        return new_embedding

    def forward(
        self,
        pixel_values: torch.Tensor,
        spatial_shapes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        target_dtype = self.patch_embedding.weight.dtype

        # pixel_values: [B, C, H, W]
        patch_size = self.patch_size

        if spatial_shapes is not None:
            # Variable-resolution: each image may have different spatial dims.
            # spatial_shapes: [B, 2]  (height, width) in pixel space
            embeddings_list = []
            position_ids_list = []
            for i in range(batch_size):
                h, w = spatial_shapes[i]
                num_patches_h = h // patch_size
                num_patches_w = w // patch_size

                # Extract patches for this image
                img = pixel_values[i : i + 1, :, :h, :w]  # [1, C, H, W]
                patches = img.unfold(2, patch_size, patch_size).unfold(
                    3, patch_size, patch_size
                )
                # patches: [1, C, num_patches_h, num_patches_w, patch_size, patch_size]
                patches = patches.contiguous().view(
                    1,
                    self.num_channels,
                    num_patches_h,
                    num_patches_w,
                    patch_size * patch_size,
                )
                patches = patches.permute(0, 2, 3, 1, 4).reshape(
                    1, num_patches_h * num_patches_w, -1
                )

                embed = self.patch_embedding(patches.to(dtype=target_dtype))
                pos_ids = torch.arange(
                    num_patches_h * num_patches_w, device=embed.device
                )
                embed = embed + self.position_embedding(pos_ids)
                embeddings_list.append(embed)
                position_ids_list.append(pos_ids)

            embeddings = torch.cat(embeddings_list, dim=0)
        else:
            # Fixed-resolution: standard patch extraction
            patches = pixel_values.unfold(2, patch_size, patch_size).unfold(
                3, patch_size, patch_size
            )
            # patches: [B, C, H/patch, W/patch, patch_size, patch_size]
            num_patches_h = pixel_values.shape[2] // patch_size
            num_patches_w = pixel_values.shape[3] // patch_size
            patches = patches.contiguous().view(
                batch_size,
                self.num_channels,
                num_patches_h,
                num_patches_w,
                patch_size * patch_size,
            )
            patches = patches.permute(0, 2, 3, 1, 4).reshape(
                batch_size, num_patches_h * num_patches_w, -1
            )

            embeddings = self.patch_embedding(patches.to(dtype=target_dtype))
            position_ids = torch.arange(
                self.num_patches, device=embeddings.device
            ).unsqueeze(0)
            embeddings = embeddings + self.position_embedding(position_ids)

        return embeddings


# ---------------------------------------------------------------------------
# SigLIP2 Attention
# ---------------------------------------------------------------------------


class Siglip2Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper."""

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                "embed_dim must be divisible by num_heads "
                f"(got `embed_dim`: {self.embed_dim} and `num_heads`: "
                f"{self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout if hasattr(config, "attention_dropout") else 0.0

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Input shape: Batch x Time x Channel"""
        bsz, tgt_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = self._shape(query_states, tgt_len, bsz)
        key_states = self._shape(key_states, tgt_len, bsz)
        value_states = self._shape(value_states, tgt_len, bsz)

        # Manual attention (eager mode)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Upcast to float32 for numerical stability
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)

        if self.dropout > 0.0 and self.training:
            attn_weights = nn.functional.dropout(
                attn_weights, p=self.dropout, training=True
            )

        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


class Siglip2SdpaAttention(Siglip2Attention):
    """SigLIP2 attention using torch F.scaled_dot_product_attention."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        bsz, tgt_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = self._shape(query_states, tgt_len, bsz)
        key_states = self._shape(key_states, tgt_len, bsz)
        value_states = self._shape(value_states, tgt_len, bsz)

        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            scale=self.scale,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output, None


# ---------------------------------------------------------------------------
# SigLIP2 MLP
# ---------------------------------------------------------------------------


class Siglip2MLP(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.activation_fn = get_act_fn(config.hidden_act)
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


# ---------------------------------------------------------------------------
# SigLIP2 Encoder Layer
# ---------------------------------------------------------------------------


class Siglip2EncoderLayer(nn.Module):
    """Pre-norm transformer layer: LN -> SelfAttn -> Residual -> LN -> MLP -> Residual."""

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.self_attn = Siglip2Attention(config)
        self.layer_norm1 = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.mlp = Siglip2MLP(config)
        self.layer_norm2 = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Pre-norm self-attention
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
        )
        hidden_states = residual + hidden_states

        # Pre-norm MLP
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


# ---------------------------------------------------------------------------
# SigLIP2 Encoder
# ---------------------------------------------------------------------------


class Siglip2Encoder(nn.Module):
    """Transformer encoder consisting of config.num_hidden_layers self-attention
    layers. Each layer is a Siglip2EncoderLayer."""

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [Siglip2EncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.gradient_checkpointing = False

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool | None = None,
        output_attentions: bool | None = None,
    ):
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else getattr(self.config, "output_hidden_states", False)
        )
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else getattr(self.config, "output_attentions", False)
        )

        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for encoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    encoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                )

            hidden_states = layer_outputs

            if output_attentions:
                # Eager attention returns (output, weights); SDPA returns (output, None)
                # We rely on the layer returning just hidden_states for now,
                # but the Siglip2EncoderLayer.forward returns a tensor, not a tuple.
                pass

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # Return a simple namespace-like object for compatibility
        return _EncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )

    @staticmethod
    def _gradient_checkpointing_func(func, *args):
        return torch.utils.checkpoint.checkpoint(func, *args, use_reentrant=False)


class _EncoderOutput:
    """Simple namespace to hold encoder outputs, replacing diffusers
    BaseModelOutput."""

    __slots__ = ("last_hidden_state", "hidden_states", "attentions")

    def __init__(self, last_hidden_state=None, hidden_states=None, attentions=None):
        self.last_hidden_state = last_hidden_state
        self.hidden_states = hidden_states
        self.attentions = attentions


# ---------------------------------------------------------------------------
# SigLIP2 Multihead Attention Pooling Head
# ---------------------------------------------------------------------------


class Siglip2MultiheadAttentionPoolingHead(nn.Module):
    """Multihead attention pooling head used for pooler_output."""

    def __init__(self, config: Config) -> None:
        super().__init__()

        self.probe = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            batch_first=True,
        )
        self.layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.mlp = Siglip2MLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        probe = self.probe.repeat(batch_size, 1, 1)

        if attention_mask is not None:
            # Convert 2D mask [B, S] to the format expected by nn.MultiheadAttention:
            # key_padding_mask: [B, S] where True = ignore
            key_padding_mask = (attention_mask == 0)
        else:
            key_padding_mask = None

        hidden_states = self.layernorm(hidden_states)
        pooler_output, _ = self.attention(
            probe,
            hidden_states,
            hidden_states,
            key_padding_mask=key_padding_mask,
        )
        pooler_output = pooler_output + self.mlp(self.layernorm(pooler_output))
        return pooler_output


# ---------------------------------------------------------------------------
# SigLIP2 Vision Transformer  -->  ImageEncoder
# ---------------------------------------------------------------------------

# Default architecture parameters for SigLIP2 (HunyuanImage-3.0)
_SIGLIP2_DEFAULT_CONFIG = {
    "hidden_size": 1152,
    "num_attention_heads": 16,
    "num_hidden_layers": 27,
    "intermediate_size": 4304,
    "patch_size": 14,
    "num_channels": 3,
    "num_patches": 729,  # 27 * 27
    "image_size": 378,
    "layer_norm_eps": 1e-6,
    "hidden_act": "gelu_pytorch_tanh",
    "attention_dropout": 0.0,
    "output_attentions": False,
    "output_hidden_states": False,
    "use_return_dict": True,
    "vision_use_head": True,
    "_attn_implementation": "eager",
}


class Siglip2VisionTransformer(ImageEncoder):
    """SigLIP-2 vision encoder for HunyuanImage-3.0, adapted to sglang's
    ImageEncoder base class."""

    config_class = ImageEncoderConfig
    main_input_name = "pixel_values"

    _supported_attention_backends: set[AttentionBackendEnum] = {
        AttentionBackendEnum.TORCH_SDPA,
    }

    def __init__(
        self,
        config: ImageEncoderConfig,
        hf_config: dict | None = None,
    ) -> None:
        super().__init__(config)

        # Build the internal vision config from HF config or defaults
        if hf_config is not None:
            self.vision_config = Config(hf_config)
        else:
            self.vision_config = Config(_SIGLIP2_DEFAULT_CONFIG.copy())

        self.embeddings = Siglip2VisionEmbeddings(self.vision_config)
        self.encoder = Siglip2Encoder(self.vision_config)
        self.post_layernorm = nn.LayerNorm(
            self.vision_config.hidden_size,
            eps=self.vision_config.layer_norm_eps,
        )

        self.use_head = True
        if hasattr(self.vision_config, "vision_use_head"):
            self.use_head = self.vision_config.vision_use_head

        if self.use_head:
            self.head = Siglip2MultiheadAttentionPoolingHead(self.vision_config)

    def forward(
        self,
        pixel_values: torch.Tensor,
        **kwargs,
    ) -> BaseEncoderOutput:
        attention_mask = kwargs.get("attention_mask", None)
        spatial_shapes = kwargs.get("spatial_shapes", None)
        output_hidden_states = kwargs.get("output_hidden_states", None)
        output_attentions = kwargs.get("output_attentions", None)

        if output_hidden_states is None:
            output_hidden_states = getattr(
                self.vision_config, "output_hidden_states", False
            )
        if output_attentions is None:
            output_attentions = getattr(
                self.vision_config, "output_attentions", False
            )

        hidden_states = self.embeddings(
            pixel_values, spatial_shapes=spatial_shapes
        )

        # Prepare 4D attention mask if a 2D mask is provided
        if attention_mask is not None and attention_mask.dim() == 2:
            encoder_attention_mask = _prepare_4d_attention_mask(
                attention_mask, hidden_states.dtype
            )
        else:
            encoder_attention_mask = attention_mask

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=encoder_attention_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )

        last_hidden_state = self.post_layernorm(
            encoder_outputs.last_hidden_state
        )

        # Compute pooler output
        pooler_output = None
        if self.use_head:
            pooler_output = self.head(
                last_hidden_state, attention_mask=attention_mask
            )
            # Squeeze the probe dimension: [B, 1, D] -> [B, D]
            pooler_output = pooler_output.squeeze(1)

        return BaseEncoderOutput(
            last_hidden_state=last_hidden_state,
            pooler_output=pooler_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> set[str]:
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            if name in params_dict:
                param = params_dict[name]
                weight_loader = getattr(
                    param, "weight_loader", default_weight_loader
                )
                weight_loader(param, loaded_weight)
                loaded_params.add(name)

        return loaded_params


# ---------------------------------------------------------------------------
# LightProjector (standalone nn.Module, NOT an ImageEncoder)
# ---------------------------------------------------------------------------


class LightProjector(nn.Module):
    """Lightweight projector used in HunyuanImage-3.0 to project vision
    encoder outputs into the text embedding space."""

    def __init__(self, config: dict) -> None:
        config = Config(config)
        super().__init__()

        if config.projector_type == "linear":
            self.layers = nn.Linear(config.input_dim, config.n_embed)
        elif config.projector_type == "mlp_gelu":
            modules = [nn.Linear(config.input_dim, config.n_embed)]
            for _ in range(1, config.depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(config.n_embed, config.n_embed))
            self.layers = nn.Sequential(*modules)
        else:
            raise ValueError(f"Unknown projector type: {config.projector_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


# ---------------------------------------------------------------------------
# Entry class for model registry
# ---------------------------------------------------------------------------

EntryClass = [Siglip2VisionTransformer]