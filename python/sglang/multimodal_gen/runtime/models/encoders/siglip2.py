# SPDX-License-Identifier: Apache-2.0
"""SigLIP2 adapter for HunyuanImage-3.0 diffusion pipelines.

The optimized SigLIP2 implementation lives in ``sglang.srt.models.siglip2``.
This module keeps the multimodal_gen ImageEncoder-facing API while delegating
the vision tower implementation and HF weight mapping to srt.
"""

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
from transformers import Siglip2VisionConfig

from sglang.multimodal_gen.configs.models.encoders import (
    BaseEncoderOutput,
    ImageEncoderConfig,
)
from sglang.multimodal_gen.runtime.models.encoders.base import ImageEncoder
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.srt.models.siglip2 import Siglip2Model

logger = init_logger(__name__)


def _to_siglip2_config(
    config: dict[str, Any] | Siglip2VisionConfig,
) -> Siglip2VisionConfig:
    if isinstance(config, Siglip2VisionConfig):
        return config
    return Siglip2VisionConfig(**config)


class Siglip2VisionTransformer(ImageEncoder):
    """multimodal_gen adapter around srt's optimized SigLIP2 model.

    ``Siglip2ImageProcessorFast`` returns padded patch tensors plus
    ``spatial_shapes`` and ``pixel_attention_mask``.  The srt SigLIP2 model uses
    packed NaFlex inputs with ``cu_seqlens``.  This adapter performs that small
    conversion and returns ``BaseEncoderOutput`` for the diffusion pipeline.
    """

    config_class = ImageEncoderConfig
    main_input_name = "pixel_values"
    _supported_attention_backends: set[AttentionBackendEnum] = {
        AttentionBackendEnum.TORCH_SDPA,
    }

    def __init__(
        self,
        config: ImageEncoderConfig | dict[str, Any] | Siglip2VisionConfig | None = None,
        hf_config: dict[str, Any] | Siglip2VisionConfig | None = None,
    ) -> None:
        encoder_config = config if isinstance(config, ImageEncoderConfig) else None
        super().__init__(encoder_config or ImageEncoderConfig())

        vision_config = hf_config if hf_config is not None else config
        if vision_config is None or isinstance(vision_config, ImageEncoderConfig):
            raise ValueError("Siglip2VisionTransformer requires a SigLIP2 HF config.")

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
        if spatial_shapes.dim() == 1:
            spatial_shapes = spatial_shapes.unsqueeze(0)
        if spatial_shapes.dim() == 3 and spatial_shapes.shape[0] == 1:
            spatial_shapes = spatial_shapes.squeeze(0)
        if spatial_shapes.dim() != 2 or spatial_shapes.shape[-1] != 2:
            raise ValueError(
                "SigLIP2 spatial_shapes must have shape (batch, 2), got "
                f"{tuple(spatial_shapes.shape)}"
            )
        return spatial_shapes.detach().to(device="cpu", dtype=torch.long)

    @staticmethod
    def _pack_pixel_values(
        pixel_values: torch.Tensor,
        spatial_shapes_cpu: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if pixel_values.dim() == 2:
            pixel_values = pixel_values.unsqueeze(0)
        if pixel_values.dim() != 3:
            raise ValueError(
                "SigLIP2 expects patchified pixel_values with shape "
                f"(batch, seq, dim), got {tuple(pixel_values.shape)}"
            )

        lengths = (spatial_shapes_cpu[:, 0] * spatial_shapes_cpu[:, 1]).tolist()
        batch_size = len(lengths)
        if pixel_values.shape[0] != batch_size:
            raise ValueError(
                "pixel_values batch size does not match spatial_shapes: "
                f"{pixel_values.shape[0]} vs {batch_size}"
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
                        "SigLIP2 attention_mask batch size does not match "
                        f"pixel_values: {attention_mask.shape[0]} vs {batch_size}"
                    )
                attention_mask = attention_mask.flatten().unsqueeze(0)
            attention_mask = attention_mask.to(
                device=pixel_values.device, dtype=torch.bool
            )

        packed_items: list[torch.Tensor] = []
        for idx, length in enumerate(lengths):
            if attention_mask is None:
                packed = pixel_values[idx, :length]
            else:
                packed = pixel_values[idx][attention_mask[idx]]
                if packed.shape[0] != length:
                    logger.warning(
                        "SigLIP2 attention_mask selected %d token(s), expected %d "
                        "from spatial_shapes for item %d.",
                        packed.shape[0],
                        length,
                        idx,
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

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        def normalize():
            for name, tensor in weights:
                if not name.startswith("vision_model."):
                    name = f"vision_model.{name}"
                yield name, tensor

        return self.model.load_weights(normalize())


class Config:
    """Small dict-to-attribute wrapper used by LightProjector."""

    def __init__(self, config: dict[str, Any]):
        for key, value in config.items():
            setattr(self, key, value)


class LightProjector(nn.Module):
    """Project SigLIP2 outputs into HunyuanImage3's text embedding space."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        config = Config(config)

        if config.projector_type == "linear":
            self.layers = nn.Linear(config.input_dim, config.n_embed)
        elif config.projector_type == "mlp_gelu":
            modules: list[nn.Module] = [nn.Linear(config.input_dim, config.n_embed)]
            for _ in range(1, config.depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(config.n_embed, config.n_embed))
            self.layers = nn.Sequential(*modules)
        else:
            raise ValueError(f"Unknown projector type: {config.projector_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


EntryClass = [Siglip2VisionTransformer]
