# SPDX-License-Identifier: Apache-2.0
"""SigLIP2 encoder implementation for HunyuanImage-3.0 diffusion pipelines.

This is a standalone implementation that does NOT depend on sglang.srt,
avoiding the sgl_kernel_npu import chain that causes triton.language.extra.cann
compatibility issues.

The implementation follows the same pattern as other multimodal_gen encoders
(CLIP, Qwen2VL, etc.) and uses multimodal_gen's own layers and attention backends.
"""

from collections.abc import Iterable
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Siglip2VisionConfig

from sglang.multimodal_gen.configs.models.encoders import (
    BaseEncoderOutput,
    ImageEncoderConfig,
)
from sglang.multimodal_gen.runtime.layers.activation import get_act_fn
from sglang.multimodal_gen.runtime.layers.linear import (
    RowParallelLinear,
)
from sglang.multimodal_gen.runtime.layers.quantization import QuantizationConfig
from sglang.multimodal_gen.runtime.loader.weight_utils import default_weight_loader
from sglang.multimodal_gen.runtime.models.encoders.base import ImageEncoder
from sglang.multimodal_gen.runtime.models.encoders.vision import (
    resolve_visual_encoder_outputs,
)
from sglang.multimodal_gen.runtime.platforms import (
    AttentionBackendEnum,
    current_platform,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


def _to_siglip2_config(
    config: dict,
) -> Siglip2VisionConfig:
    """Convert dict config to Siglip2VisionConfig."""
    if isinstance(config, Siglip2VisionConfig):
        return config
    # Filter out keys not recognised by Siglip2VisionConfig
    default_config = Siglip2VisionConfig()
    valid_fields = set(default_config.to_dict().keys())

    # Also filter out known extra fields that appear in HunyuanImage3 config
    extra_fields_to_remove = {
        '_attn_implementation', 'torch_dtype', 'use_return_dict'
    }

    filtered = {k: v for k, v in config.items()
                if k in valid_fields and k not in extra_fields_to_remove}

    return Siglip2VisionConfig(**filtered)


class Siglip2VisionEmbeddings(nn.Module):
    """Siglip2 vision embeddings with NaFlex variable-resolution support."""

    def __init__(self, config: Siglip2VisionConfig | dict):
        super().__init__()
        # Convert dict to Siglip2VisionConfig if needed
        if isinstance(config, dict):
            config = _to_siglip2_config(config)

        self.config = config
        self.embed_dim = config.hidden_size
        self.patch_size = config.patch_size

        # Siglip2 uses Linear instead of Conv2d for patch embedding
        self.patch_embedding = nn.Linear(
            in_features=config.num_channels * self.patch_size * self.patch_size,
            out_features=self.embed_dim,
        )
        self.num_patches = config.num_patches
        self.position_embedding_size = int(self.num_patches**0.5)
        self.position_embedding = nn.Embedding(self.num_patches, self.embed_dim)

    def forward(
        self,
        pixel_values_packed: torch.FloatTensor,
        spatial_shapes: torch.LongTensor,
    ) -> torch.Tensor:
        """Embed patchified pixel values in packed (unpadded) form.

        Args:
            pixel_values_packed: (1, total_tokens, patch_dim) or
                (total_tokens, patch_dim), packed in tile order.
            spatial_shapes: (num_tiles, 2) on CPU (height, width) per tile.

        Returns:
            (1, total_tokens, embed_dim) packed embeddings.
        """
        assert spatial_shapes.device.type == "cpu", (
            "Expected `spatial_shapes` on CPU to avoid device-to-host sync."
        )

        if pixel_values_packed.dim() == 3:
            assert pixel_values_packed.shape[0] == 1
            pixel_values_flat = pixel_values_packed[0]
        else:
            pixel_values_flat = pixel_values_packed

        lengths = (spatial_shapes[:, 0] * spatial_shapes[:, 1]).to(dtype=torch.int64)
        lengths_list = lengths.tolist()
        total_tokens = int(sum(lengths_list))
        if total_tokens != pixel_values_flat.shape[0]:
            raise ValueError(
                "Packed pixel_values token count does not match spatial_shapes: "
                f"{pixel_values_flat.shape[0]} vs {total_tokens}."
            )

        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values_flat.to(dtype=target_dtype))

        positional_embeddings = self.position_embedding.weight.reshape(
            self.position_embedding_size, self.position_embedding_size, -1
        )
        packed_pos_embeds = self.resize_positional_embeddings_packed(
            positional_embeddings,
            spatial_shapes,
            lengths_list=lengths_list,
        )

        embeddings = patch_embeds + packed_pos_embeds
        return embeddings.unsqueeze(0)

    @staticmethod
    def resize_positional_embeddings_packed(
        positional_embeddings: torch.Tensor,
        spatial_shapes: torch.LongTensor,
        lengths_list: list,
    ) -> torch.Tensor:
        """Resize positional embeddings per image and return a packed tensor.

        Args:
            positional_embeddings: (height, width, embed_dim) base grid.
            spatial_shapes: (batch_size, 2) on CPU, (height, width) per image.
            lengths_list: flattened token length per image (height * width).

        Returns:
            (total_tokens, embed_dim) packed positional embeddings.
        """
        assert spatial_shapes.device.type == "cpu"

        embed_dim = positional_embeddings.shape[-1]
        source_dtype = positional_embeddings.dtype

        total_tokens = int(sum(lengths_list))
        packed_pos_embeds = torch.empty(
            (total_tokens, embed_dim),
            device=positional_embeddings.device,
            dtype=source_dtype,
        )

        # (height, width, embed_dim) -> (1, embed_dim, height, width)
        pos_4d = positional_embeddings.permute(2, 0, 1).unsqueeze(0)

        # Upcast to float32 on CPU for antialias support
        if pos_4d.device.type == "cpu":
            pos_4d = pos_4d.to(torch.float32)

        offset = 0
        for i, length in enumerate(lengths_list):
            if length <= 0:
                continue
            height, width = spatial_shapes[i].tolist()
            resized = F.interpolate(
                pos_4d,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            resized = resized.reshape(embed_dim, height * width).transpose(0, 1)
            resized = resized.to(source_dtype)
            packed_pos_embeds[offset : offset + length] = resized
            offset += length

        return packed_pos_embeds


class Siglip2Attention(nn.Module):
    """Multi-headed attention for Siglip2 with variable-length sequence support."""

    def __init__(
        self,
        config: Siglip2VisionConfig | dict,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        # Convert dict to Siglip2VisionConfig if needed
        if isinstance(config, dict):
            config = _to_siglip2_config(config)

        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads

        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads "
                f"(got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
            )

        self.scale = self.head_dim**-0.5

        # Use standard nn.Linear for vision encoder - completely independent of TP
        # Vision encoder weights should NOT be split across TP ranks
        # Separate Q, K, V projections (not fused) to match checkpoint weight names
        self.q_proj = nn.Linear(
            in_features=self.embed_dim,
            out_features=self.embed_dim,
            bias=True,
        )
        self.k_proj = nn.Linear(
            in_features=self.embed_dim,
            out_features=self.embed_dim,
            bias=True,
        )
        self.v_proj = nn.Linear(
            in_features=self.embed_dim,
            out_features=self.embed_dim,
            bias=True,
        )

        # Output projection
        self.out_proj = nn.Linear(
            in_features=self.embed_dim,
            out_features=self.embed_dim,
            bias=True,
        )

        # Use SDPA for variable-length attention
        self._supported_attention_backends = {AttentionBackendEnum.TORCH_SDPA}

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int | torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass with variable-length attention.

        Args:
            hidden_states: (1, total_tokens, embed_dim) packed hidden states
            cu_seqlens: Cumulative sequence lengths for variable-length attention
            max_seqlen: Maximum sequence length

        Returns:
            (1, total_tokens, embed_dim) attention output
        """
        # hidden_states: (1, total_tokens, embed_dim)
        if hidden_states.dim() == 3 and hidden_states.shape[0] == 1:
            hidden_states = hidden_states.squeeze(0)  # (total_tokens, embed_dim)

        # Separate Q, K, V projections
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape for attention: (total_tokens, num_heads, head_dim)
        query_states = query_states.view(-1, self.num_heads, self.head_dim)
        key_states = key_states.view(-1, self.num_heads, self.head_dim)
        value_states = value_states.view(-1, self.num_heads, self.head_dim)

        # Use torch SDPA with variable-length sequence
        # Need to expand to batch dimension based on cu_seqlens
        batch_size = cu_seqlens.shape[0] - 1
        query_states = query_states.unsqueeze(0)  # (1, total_tokens, num_heads, head_dim)
        key_states = key_states.unsqueeze(0)
        value_states = value_states.unsqueeze(0)

        # For variable-length, we use a simple approach: process each sequence separately
        # This is not optimal but works for the multimodal_gen use case
        outputs = []
        for i in range(batch_size):
            start_idx = cu_seqlens[i].item()
            end_idx = cu_seqlens[i + 1].item()

            q_i = query_states[:, start_idx:end_idx]  # (1, seq_len_i, num_heads, head_dim)
            k_i = key_states[:, start_idx:end_idx]
            v_i = value_states[:, start_idx:end_idx]

            # SDPA expects (batch, seq_len, num_heads, head_dim) or (batch, num_heads, seq_len, head_dim)
            # Transpose to (batch, num_heads, seq_len, head_dim)
            q_i = q_i.transpose(1, 2)
            k_i = k_i.transpose(1, 2)
            v_i = v_i.transpose(1, 2)

            # Use scaled_dot_product_attention
            attn_output_i = torch.nn.functional.scaled_dot_product_attention(
                q_i, k_i, v_i,
                is_causal=False,
            )

            # Transpose back: (1, seq_len_i, num_heads, head_dim)
            attn_output_i = attn_output_i.transpose(1, 2)
            outputs.append(attn_output_i)

        # Concatenate all sequences
        attn_output = torch.cat(outputs, dim=1)  # (1, total_tokens, num_heads, head_dim)
        attn_output = attn_output.view(1, -1, self.embed_dim)  # (1, total_tokens, embed_dim)

        # Output projection
        attn_output = self.out_proj(attn_output)

        return attn_output


class Siglip2MLP(nn.Module):
    """MLP for Siglip2 encoder layers."""

    def __init__(
        self,
        config: Siglip2VisionConfig | dict,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        # Convert dict to Siglip2VisionConfig if needed
        if isinstance(config, dict):
            config = _to_siglip2_config(config)

        self.config = config
        self.activation_fn = get_act_fn(config.hidden_act)

        # Use standard nn.Linear for vision encoder - completely independent of TP
        # Vision encoder weights should NOT be split across TP ranks
        self.fc1 = nn.Linear(
            in_features=config.hidden_size,
            out_features=config.intermediate_size,
            bias=True,
        )
        self.fc2 = nn.Linear(
            in_features=config.intermediate_size,
            out_features=config.hidden_size,
            bias=True,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class Siglip2EncoderLayer(nn.Module):
    """Single encoder layer for Siglip2."""

    def __init__(
        self,
        config: Siglip2VisionConfig | dict,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        # Convert dict to Siglip2VisionConfig if needed
        if isinstance(config, dict):
            config = _to_siglip2_config(config)

        self.embed_dim = config.hidden_size
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.self_attn = Siglip2Attention(
            config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = Siglip2MLP(
            config,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int | torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for encoder layer."""
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class Siglip2Encoder(nn.Module):
    """Transformer encoder for Siglip2."""

    def __init__(
        self,
        config: Siglip2VisionConfig | dict,
        quant_config: Optional[QuantizationConfig] = None,
        num_hidden_layers_override: Optional[int] = None,
        prefix: str = "",
    ):
        super().__init__()
        # Convert dict to Siglip2VisionConfig if needed
        if isinstance(config, dict):
            config = _to_siglip2_config(config)

        self.config = config

        if num_hidden_layers_override is None:
            num_hidden_layers = config.num_hidden_layers
        else:
            num_hidden_layers = num_hidden_layers_override

        self.layers = nn.ModuleList(
            [
                Siglip2EncoderLayer(
                    config=config,
                    quant_config=quant_config,
                    prefix=f"{prefix}.layers.{idx}",
                )
                for idx in range(num_hidden_layers)
            ]
        )

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int | torch.Tensor,
        return_all_hidden_states: bool = False,
    ) -> torch.Tensor | list:
        hidden_states_pool = [inputs_embeds]
        hidden_states = inputs_embeds

        for encoder_layer in self.layers:
            hidden_states = encoder_layer(
                hidden_states,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
            if return_all_hidden_states:
                hidden_states_pool.append(hidden_states)
        if return_all_hidden_states:
            return hidden_states_pool
        return hidden_states


class Siglip2VisionTransformer(nn.Module):
    """Siglip2 Vision Transformer with NaFlex variable-resolution support."""

    def __init__(
        self,
        config: Siglip2VisionConfig | dict,
        quant_config: Optional[QuantizationConfig] = None,
        num_hidden_layers_override: Optional[int] = None,
        require_post_norm: Optional[bool] = None,
        prefix: str = "",
    ):
        super().__init__()
        # Convert dict to Siglip2VisionConfig if needed
        if isinstance(config, dict):
            config = _to_siglip2_config(config)

        embed_dim = config.hidden_size
        self.config = config
        self.embeddings = Siglip2VisionEmbeddings(config)
        self.encoder = Siglip2Encoder(
            config,
            quant_config=quant_config,
            num_hidden_layers_override=num_hidden_layers_override,
            prefix=f"{prefix}.encoder",
        )
        num_hidden_layers = config.num_hidden_layers
        if len(self.encoder.layers) > config.num_hidden_layers:
            raise ValueError(
                f"The original encoder only has {num_hidden_layers} "
                f"layers, but you requested {len(self.encoder.layers)} layers."
            )

        if require_post_norm is None:
            require_post_norm = len(self.encoder.layers) == num_hidden_layers

        if require_post_norm:
            self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        else:
            self.post_layernorm = None

    @property
    def dtype(self) -> torch.dtype:
        return self.embeddings.patch_embedding.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.embeddings.patch_embedding.weight.device

    def forward(
        self,
        pixel_values_packed: torch.FloatTensor,
        spatial_shapes: torch.LongTensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: torch.Tensor,
        select_layers: Optional[list] = None,
    ) -> torch.Tensor:
        """Forward pass through the vision transformer."""
        hidden_states = self.embeddings(pixel_values_packed, spatial_shapes)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            return_all_hidden_states=select_layers is not None,
        )

        encoder_outputs = resolve_visual_encoder_outputs(
            encoder_outputs,
            self.post_layernorm,
            select_layers=select_layers,
            max_possible_layers=self.config.num_hidden_layers,
        )

        return encoder_outputs


class Siglip2Model(nn.Module):
    """Siglip2 Vision Model wrapper."""

    def __init__(
        self,
        config: Siglip2VisionConfig | dict,
        quant_config: Optional[QuantizationConfig] = None,
        num_hidden_layers_override: Optional[int] = None,
        require_post_norm: Optional[bool] = None,
        prefix: str = "",
    ):
        super().__init__()

        # Convert dict to Siglip2VisionConfig if needed
        if isinstance(config, dict):
            config = _to_siglip2_config(config)

        self.vision_model = Siglip2VisionTransformer(
            config,
            quant_config=quant_config,
            num_hidden_layers_override=num_hidden_layers_override,
            require_post_norm=require_post_norm,
            prefix=f"{prefix}.vision_model",
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.vision_model.dtype

    @property
    def device(self) -> torch.device:
        return self.vision_model.device

    def forward(
        self,
        pixel_values_packed: torch.FloatTensor,
        spatial_shapes: torch.LongTensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: torch.Tensor,
        select_layers: Optional[list] = None,
    ) -> torch.Tensor:
        """Forward pass through the vision model."""
        return self.vision_model(
            pixel_values_packed=pixel_values_packed,
            spatial_shapes=spatial_shapes,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            select_layers=select_layers,
        )

    def load_weights(self, weights: Iterable) -> set:
        """Load weights with proper handling of fused QKV and layer selection."""
        # No stacked params - we use separate q/k/v projections matching checkpoint
        params_dict = dict(self.named_parameters())
        loaded_params = set()
        layer_count = len(self.vision_model.encoder.layers)

        for name, loaded_weight in weights:
            # post_layernorm is optional
            if (
                name.startswith("post_layernorm")
                and self.vision_model.post_layernorm is None
            ):
                continue

            # Skip layers beyond override count
            if name.startswith("encoder.layers"):
                parts = name.split(".")
                if len(parts) >= 2:
                    layer_idx = int(parts[1])
                    if layer_idx >= layer_count:
                        continue

            if name not in params_dict:
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)

            loaded_params.add(name)

        return loaded_params


class Siglip2VisionTransformerEncoder(ImageEncoder):
    """multimodal_gen ImageEncoder adapter for Siglip2.

    This adapter handles the conversion from ImageEncoder API to Siglip2Model.
    """

    config_class = ImageEncoderConfig
    main_input_name = "pixel_values"
    _supported_attention_backends = {AttentionBackendEnum.TORCH_SDPA}

    def __init__(
        self,
        config: ImageEncoderConfig | dict | Siglip2VisionConfig | None = None,
        hf_config: dict | Siglip2VisionConfig | None = None,
    ) -> None:
        encoder_config = config if isinstance(config, ImageEncoderConfig) else None
        super().__init__(encoder_config or ImageEncoderConfig())

        vision_config = hf_config if hf_config is not None else config
        if vision_config is None or isinstance(vision_config, ImageEncoderConfig):
            raise ValueError("Siglip2VisionTransformerEncoder requires a SigLIP2 HF config.")

        self.vision_config = _to_siglip2_config(vision_config)
        self.model = Siglip2Model(self.vision_config)

    @property
    def device(self) -> torch.device:
        return self.model.device

    @property
    def dtype(self) -> torch.dtype:
        return self.model.dtype

    @staticmethod
    def _normalize_spatial_shapes(spatial_shapes: torch.Tensor) -> torch.Tensor:
        """Normalize spatial_shapes to (batch, 2) format."""
        if spatial_shapes.dim() == 1:
            spatial_shapes = spatial_shapes.unsqueeze(0)
        if spatial_shapes.dim() == 3 and spatial_shapes.shape[0] == 1:
            spatial_shapes = spatial_shapes.squeeze(0)
        if spatial_shapes.dim() != 2 or spatial_shapes.shape[-1] != 2:
            raise ValueError(
                f"spatial_shapes must have shape (batch, 2), got {tuple(spatial_shapes.shape)}"
            )
        return spatial_shapes.detach().to(device="cpu", dtype=torch.long)

    @staticmethod
    def _pack_pixel_values(
        pixel_values: torch.Tensor,
        spatial_shapes_cpu: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> tuple:
        """Pack pixel_values into variable-length format."""
        if pixel_values.dim() == 2:
            pixel_values = pixel_values.unsqueeze(0)
        if pixel_values.dim() != 3:
            raise ValueError(
                f"pixel_values must be (batch, seq, dim), got {tuple(pixel_values.shape)}"
            )

        lengths = (spatial_shapes_cpu[:, 0] * spatial_shapes_cpu[:, 1]).tolist()
        batch_size = len(lengths)
        if pixel_values.shape[0] != batch_size:
            raise ValueError(
                f"pixel_values batch size mismatch: {pixel_values.shape[0]} vs {batch_size}"
            )

        if attention_mask is not None:
            if attention_mask.dim() == 1:
                attention_mask = attention_mask.unsqueeze(0)
            if attention_mask.dim() == 3 and attention_mask.shape[0] == 1:
                attention_mask = attention_mask.squeeze(0)
            if attention_mask.dim() > 2:
                attention_mask = attention_mask.flatten(start_dim=1)
            if attention_mask.dim() == 2 and attention_mask.shape[0] != batch_size:
                if batch_size != 1:
                    raise ValueError(
                        f"attention_mask batch size mismatch: {attention_mask.shape[0]} vs {batch_size}"
                    )
                attention_mask = attention_mask.flatten().unsqueeze(0)
            attention_mask = attention_mask.to(
                device=pixel_values.device, dtype=torch.bool
            )

        packed_items = []
        for idx, length in enumerate(lengths):
            if attention_mask is None:
                packed = pixel_values[idx, :length]
            else:
                packed = pixel_values[idx][attention_mask[idx]]
                if packed.shape[0] != length:
                    logger.warning(
                        f"attention_mask selected {packed.shape[0]} tokens, expected {length} for item {idx}"
                    )
            packed_items.append(packed)

        packed_pixel_values = torch.cat(packed_items, dim=0).unsqueeze(0)
        cu_seqlens = torch.tensor(
            [0] + torch.tensor(lengths, dtype=torch.int32).cumsum(0).tolist(),
            device=pixel_values.device,
            dtype=torch.int32,
        )
        max_seqlen = torch.tensor(max(lengths), device=pixel_values.device)
        return packed_pixel_values, cu_seqlens, max_seqlen

    def forward(self, pixel_values: torch.Tensor, **kwargs) -> BaseEncoderOutput:
        """Forward pass through Siglip2 encoder."""
        spatial_shapes = kwargs.get("spatial_shapes")
        if spatial_shapes is None:
            raise ValueError("SigLIP2 forward requires spatial_shapes.")

        spatial_shapes_cpu = self._normalize_spatial_shapes(spatial_shapes)
        packed_pixel_values, cu_seqlens, max_seqlen = self._pack_pixel_values(
            pixel_values,
            spatial_shapes_cpu,
            kwargs.get("attention_mask"),
        )
        last_hidden_state = self.model(
            pixel_values_packed=packed_pixel_values,
            spatial_shapes=spatial_shapes_cpu,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            select_layers=kwargs.get("select_layers"),
        )
        return BaseEncoderOutput(last_hidden_state=last_hidden_state)

    def load_weights(self, weights: Iterable) -> set:
        """Load weights with vision_model prefix handling."""
        def normalize():
            for name, tensor in weights:
                if not name.startswith("vision_model."):
                    name = f"vision_model.{name}"
                yield name, tensor

        return self.model.load_weights(normalize())


class Config:
    """Small dict-to-attribute wrapper used by LightProjector."""

    def __init__(self, config: dict):
        for key, value in config.items():
            setattr(self, key, value)


class LightProjector(nn.Module):
    """Project SigLIP2 outputs into HunyuanImage3's text embedding space."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        config = Config(config)

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


EntryClass = [Siglip2VisionTransformerEncoder]