# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo
# Ported from vllm-omni's vllm_omni/diffusion/models/hunyuan_image3/autoencoder.py

# SPDX-License-Identifier: Apache-2.0

import math

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn

from sglang.multimodal_gen.configs.models.vaes.hunyuan_image3 import (
    HunyuanImage3VAEConfig,
)
from sglang.multimodal_gen.runtime.models.vaes.common import ParallelTiledVAE


def swish(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


def forward_with_checkpointing(module, *inputs, use_checkpointing=False):
    def create_custom_forward(module):
        def custom_forward(*inputs):
            return module(*inputs)

        return custom_forward

    if use_checkpointing:
        return torch.utils.checkpoint.checkpoint(
            create_custom_forward(module), *inputs, use_reentrant=False
        )
    else:
        return module(*inputs)


class Conv3d(nn.Conv3d):
    """Perform Conv3d on patches with numerical differences from nn.Conv3d within 1e-5.
    Only symmetric padding is supported."""

    def forward(self, input):
        B, C, T, H, W = input.shape
        memory_count = (C * T * H * W) * 2 / 1024**3
        if memory_count > 2:
            n_split = math.ceil(memory_count / 2)
            assert n_split >= 2
            chunks = torch.chunk(input, chunks=n_split, dim=-3)
            padded_chunks = []
            for i in range(len(chunks)):
                if self.padding[0] > 0:
                    padded_chunk = F.pad(
                        chunks[i],
                        (0, 0, 0, 0, self.padding[0], self.padding[0]),
                        mode="constant"
                        if self.padding_mode == "zeros"
                        else self.padding_mode,
                        value=0,
                    )
                    if i > 0:
                        padded_chunk[:, :, : self.padding[0]] = chunks[i - 1][
                            :, :, -self.padding[0] :
                        ]
                    if i < len(chunks) - 1:
                        padded_chunk[:, :, -self.padding[0] :] = chunks[i + 1][
                            :, :, : self.padding[0]
                        ]
                else:
                    padded_chunk = chunks[i]
                padded_chunks.append(padded_chunk)
            padding_bak = self.padding
            self.padding = (0, self.padding[1], self.padding[2])
            outputs = []
            for i in range(len(padded_chunks)):
                outputs.append(super().forward(padded_chunks[i]))
            self.padding = padding_bak
            return torch.cat(outputs, dim=-3)
        else:
            return super().forward(input)


class AttnBlock(nn.Module):
    """Attention with torch sdpa implementation."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.norm = nn.GroupNorm(
            num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
        )
        self.q = Conv3d(in_channels, in_channels, kernel_size=1)
        self.k = Conv3d(in_channels, in_channels, kernel_size=1)
        self.v = Conv3d(in_channels, in_channels, kernel_size=1)
        self.proj_out = Conv3d(in_channels, in_channels, kernel_size=1)

    def attention(self, h_: Tensor) -> Tensor:
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)
        b, c, f, h, w = q.shape
        q = rearrange(q, "b c f h w -> b 1 (f h w) c").contiguous()
        k = rearrange(k, "b c f h w -> b 1 (f h w) c").contiguous()
        v = rearrange(v, "b c f h w -> b 1 (f h w) c").contiguous()
        h_ = nn.functional.scaled_dot_product_attention(q, k, v)
        return rearrange(
            h_, "b 1 (f h w) c -> b c f h w", f=f, h=h, w=w, c=c, b=b
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.proj_out(self.attention(x))


class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.norm1 = nn.GroupNorm(
            num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
        )
        self.conv1 = Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(
            num_groups=32, num_channels=out_channels, eps=1e-6, affine=True
        )
        self.conv2 = Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            self.nin_shortcut = Conv3d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0
            )

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = swish(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = swish(h)
        h = self.conv2(h)
        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        return x + h


class DownsampleDCAE(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, add_temporal_downsample: bool = True
    ):
        super().__init__()
        factor = 2 * 2 * 2 if add_temporal_downsample else 1 * 2 * 2
        assert out_channels % factor == 0
        self.conv = Conv3d(
            in_channels, out_channels // factor, kernel_size=3, stride=1, padding=1
        )
        self.add_temporal_downsample = add_temporal_downsample
        self.group_size = factor * in_channels // out_channels

    def forward(self, x: Tensor):
        r1 = 2 if self.add_temporal_downsample else 1
        h = self.conv(x)
        h = rearrange(
            h, "b c (f r1) (h r2) (w r3) -> b (r1 r2 r3 c) f h w", r1=r1, r2=2, r3=2
        )
        shortcut = rearrange(
            x, "b c (f r1) (h r2) (w r3) -> b (r1 r2 r3 c) f h w", r1=r1, r2=2, r3=2
        )
        B, C, T, H, W = shortcut.shape
        shortcut = shortcut.view(B, h.shape[1], self.group_size, T, H, W).mean(dim=2)
        return h + shortcut


class UpsampleDCAE(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, add_temporal_upsample: bool = True
    ):
        super().__init__()
        factor = 2 * 2 * 2 if add_temporal_upsample else 1 * 2 * 2
        self.conv = Conv3d(
            in_channels, out_channels * factor, kernel_size=3, stride=1, padding=1
        )
        self.add_temporal_upsample = add_temporal_upsample
        self.repeats = factor * out_channels // in_channels

    def forward(self, x: Tensor):
        r1 = 2 if self.add_temporal_upsample else 1
        h = self.conv(x)
        h = rearrange(
            h, "b (r1 r2 r3 c) f h w -> b c (f r1) (h r2) (w r3)", r1=r1, r2=2, r3=2
        )
        shortcut = x.repeat_interleave(repeats=self.repeats, dim=1)
        shortcut = rearrange(
            shortcut,
            "b (r1 r2 r3 c) f h w -> b c (f r1) (h r2) (w r3)",
            r1=r1,
            r2=2,
            r3=2,
        )
        return h + shortcut


class Encoder(nn.Module):
    """The encoder network of AutoencoderKLConv3D."""

    def __init__(
        self,
        in_channels,
        z_channels,
        block_out_channels,
        num_res_blocks,
        ffactor_spatial,
        ffactor_temporal,
        downsample_match_channel=True,
    ):
        super().__init__()
        assert block_out_channels[-1] % (2 * z_channels) == 0
        self.z_channels = z_channels
        self.block_out_channels = block_out_channels
        self.num_res_blocks = num_res_blocks
        self.conv_in = Conv3d(
            in_channels, block_out_channels[0], kernel_size=3, stride=1, padding=1
        )
        self.down = nn.ModuleList()
        block_in = block_out_channels[0]
        for i_level, ch in enumerate(block_out_channels):
            block = nn.ModuleList()
            block_out = ch
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            down = nn.Module()
            down.block = block
            add_spatial_downsample = bool(i_level < np.log2(ffactor_spatial))
            add_temporal_downsample = add_spatial_downsample and bool(
                i_level >= np.log2(ffactor_spatial // ffactor_temporal)
            )
            if add_spatial_downsample or add_temporal_downsample:
                assert i_level < len(block_out_channels) - 1
                block_out = (
                    block_out_channels[i_level + 1]
                    if downsample_match_channel
                    else block_in
                )
                down.downsample = DownsampleDCAE(
                    block_in, block_out, add_temporal_downsample
                )
                block_in = block_out
            self.down.append(down)
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.norm_out = nn.GroupNorm(
            num_groups=32, num_channels=block_in, eps=1e-6, affine=True
        )
        self.conv_out = Conv3d(
            block_in, 2 * z_channels, kernel_size=3, stride=1, padding=1
        )
        self.gradient_checkpointing = False

    def forward(self, x: Tensor) -> Tensor:
        use_checkpointing = bool(self.training and self.gradient_checkpointing)
        h = self.conv_in(x)
        for i_level in range(len(self.block_out_channels)):
            for i_block in range(self.num_res_blocks):
                h = forward_with_checkpointing(
                    self.down[i_level].block[i_block],
                    h,
                    use_checkpointing=use_checkpointing,
                )
            if hasattr(self.down[i_level], "downsample"):
                h = forward_with_checkpointing(
                    self.down[i_level].downsample,
                    h,
                    use_checkpointing=use_checkpointing,
                )
        h = forward_with_checkpointing(
            self.mid.block_1, h, use_checkpointing=use_checkpointing
        )
        h = forward_with_checkpointing(
            self.mid.attn_1, h, use_checkpointing=use_checkpointing
        )
        h = forward_with_checkpointing(
            self.mid.block_2, h, use_checkpointing=use_checkpointing
        )
        group_size = self.block_out_channels[-1] // (2 * self.z_channels)
        shortcut = rearrange(h, "b (c r) f h w -> b c r f h w", r=group_size).mean(
            dim=2
        )
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        h += shortcut
        return h


class Decoder(nn.Module):
    """The decoder network of AutoencoderKLConv3D."""

    def __init__(
        self,
        z_channels,
        out_channels,
        block_out_channels,
        num_res_blocks,
        ffactor_spatial,
        ffactor_temporal,
        upsample_match_channel=True,
    ):
        super().__init__()
        assert block_out_channels[0] % z_channels == 0
        self.z_channels = z_channels
        self.block_out_channels = block_out_channels
        self.num_res_blocks = num_res_blocks
        block_in = block_out_channels[0]
        self.conv_in = Conv3d(
            z_channels, block_in, kernel_size=3, stride=1, padding=1
        )
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.up = nn.ModuleList()
        for i_level, ch in enumerate(block_out_channels):
            block = nn.ModuleList()
            block_out = ch
            for _ in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            up = nn.Module()
            up.block = block
            add_spatial_upsample = bool(i_level < np.log2(ffactor_spatial))
            add_temporal_upsample = bool(i_level < np.log2(ffactor_temporal))
            if add_spatial_upsample or add_temporal_upsample:
                assert i_level < len(block_out_channels) - 1
                block_out = (
                    block_out_channels[i_level + 1]
                    if upsample_match_channel
                    else block_in
                )
                up.upsample = UpsampleDCAE(
                    block_in, block_out, add_temporal_upsample
                )
                block_in = block_out
            self.up.append(up)
        self.norm_out = nn.GroupNorm(
            num_groups=32, num_channels=block_in, eps=1e-6, affine=True
        )
        self.conv_out = Conv3d(
            block_in, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.gradient_checkpointing = False

    def forward(self, z: Tensor) -> Tensor:
        use_checkpointing = bool(self.training and self.gradient_checkpointing)
        repeats = self.block_out_channels[0] // (self.z_channels)
        h = self.conv_in(z) + z.repeat_interleave(repeats=repeats, dim=1)
        h = forward_with_checkpointing(
            self.mid.block_1, h, use_checkpointing=use_checkpointing
        )
        h = forward_with_checkpointing(
            self.mid.attn_1, h, use_checkpointing=use_checkpointing
        )
        h = forward_with_checkpointing(
            self.mid.block_2, h, use_checkpointing=use_checkpointing
        )
        for i_level in range(len(self.block_out_channels)):
            for i_block in range(self.num_res_blocks + 1):
                h = forward_with_checkpointing(
                    self.up[i_level].block[i_block],
                    h,
                    use_checkpointing=use_checkpointing,
                )
            if hasattr(self.up[i_level], "upsample"):
                h = forward_with_checkpointing(
                    self.up[i_level].upsample,
                    h,
                    use_checkpointing=use_checkpointing,
                )
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        return h


class AutoencoderKLConv3D(ParallelTiledVAE):
    """VAE model for HunyuanImage-3.0 with KL loss.

    Ported from vllm-omni's AutoencoderKLConv3D but adapted to sglang's
    ParallelTiledVAE base class, which handles all tiling/slicing logic.
    """

    _aliases = ["AutoencoderKLConv3D"]
    _supports_gradient_checkpointing = True

    def __init__(self, config: HunyuanImage3VAEConfig) -> None:
        nn.Module.__init__(self)
        ParallelTiledVAE.__init__(self, config)

        arch = config.arch_config
        self.ffactor_spatial = arch.ffactor_spatial
        self.ffactor_temporal = arch.ffactor_temporal
        self.block_out_channels = arch.block_out_channels

        if config.load_encoder:
            self.encoder = Encoder(
                in_channels=arch.in_channels,
                z_channels=arch.latent_channels,
                block_out_channels=arch.block_out_channels,
                num_res_blocks=arch.layers_per_block,
                ffactor_spatial=arch.ffactor_spatial,
                ffactor_temporal=arch.ffactor_temporal,
                downsample_match_channel=arch.downsample_match_channel,
            )

        if config.load_decoder:
            self.decoder = Decoder(
                z_channels=arch.latent_channels,
                out_channels=arch.out_channels,
                block_out_channels=list(reversed(arch.block_out_channels)),
                num_res_blocks=arch.layers_per_block,
                ffactor_spatial=arch.ffactor_spatial,
                ffactor_temporal=arch.ffactor_temporal,
                upsample_match_channel=arch.upsample_match_channel,
            )

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        # x is (B, C, T, H, W) from ParallelTiledVAE.encode()
        # For single-frame inputs (HunyuanImage-3.0 image mode), expand to
        # ffactor_temporal frames so the encoder can produce valid latents.
        if x.shape[2] == 1:
            x = x.expand(-1, -1, self.ffactor_temporal, -1, -1)
        return self.encoder(x)

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        # z is (B, C, T_latent, H_latent, W_latent)
        decoded = self.decoder(z)
        # If the input was a single latent frame, take the last decoded frame
        # to get back a single output frame.
        if z.shape[2] == 1:
            decoded = decoded[:, :, -1:]
        return decoded

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (Encoder, Decoder)):
            module.gradient_checkpointing = value


EntryClass = AutoencoderKLConv3D
