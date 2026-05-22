from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.vaes.base import VAEArchConfig, VAEConfig


@dataclass
class HunyuanImage3VAEArchConfig(VAEArchConfig):
    in_channels: int = 3
    out_channels: int = 3
    latent_channels: int = 32
    block_out_channels: tuple[int, ...] = (128, 256, 512, 1024, 1024)
    layers_per_block: int = 2
    ffactor_spatial: int = 16
    ffactor_temporal: int = 4
    sample_size: int = 384
    sample_tsize: int = 96
    scaling_factor: float = 0.562679178327931
    shift_factor: float = 0.0
    spatial_compression_ratio: int = 16
    temporal_compression_ratio: int = 4
    downsample_match_channel: bool = True
    upsample_match_channel: bool = True

    def __post_init__(self):
        self.spatial_compression_ratio = self.ffactor_spatial
        self.temporal_compression_ratio = self.ffactor_temporal


@dataclass
class HunyuanImage3VAEConfig(VAEConfig):
    arch_config: VAEArchConfig = field(default_factory=HunyuanImage3VAEArchConfig)
    use_temporal_tiling: bool = False  # 图像模式不需要时间 tiling