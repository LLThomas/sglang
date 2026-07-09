# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm-omni:
#   vllm_omni/diffusion/models/hunyuan_image3/hunyuan_image3_transformer.py
#   vllm_omni/diffusion/models/hunyuan_image3/pipeline_hunyuan_image3.py
#
# This module contains the DiT (Diffusion Transformer) backbone for
# HunyuanImage-3.0, ported to sglang's CachableDiT base class.
# Pipeline / scheduler / image-processing code is NOT included here;
# it lives in the sglang pipeline framework.

import logging
import math
import re
import time
from contextlib import nullcontext
from typing import Any, Iterable

import numpy as np
import torch
from einops import rearrange
from torch import nn

from sglang.multimodal_gen.configs.models.dits.hunyuan_image3 import (
    HunyuanImage3DiTConfig,
)
from sglang.multimodal_gen.runtime.distributed import (
    get_sp_group,
    get_sp_world_size,
    sequence_model_parallel_all_gather,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_sequence_parallel_world_size,
)
from sglang.multimodal_gen.runtime.distributed.communication_op import (
    tensor_model_parallel_all_gather,
)
from sglang.multimodal_gen.runtime.layers.attention.layer import USPAttention
from sglang.multimodal_gen.runtime.layers.activation import SiluAndMul
from sglang.multimodal_gen.runtime.layers.layernorm import RMSNorm
from sglang.multimodal_gen.runtime.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.multimodal_gen.runtime.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)
from sglang.multimodal_gen.runtime.managers.forward_context import (
    get_forward_context,
    set_forward_context,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    LayerwiseOffloadableModuleMixin,
)
from sglang.multimodal_gen.runtime.models.dits.base import CachableDiT
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
)
from sglang.srt.utils import is_cuda, is_npu

try:
    import torch.distributed as dist
except ImportError:
    dist = None

logger = logging.getLogger(__name__)


def _is_rank0() -> bool:
    if dist is None or not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def zero_module(module: nn.Module) -> nn.Module:
    """Zero out the parameters of a module and return it."""
    for p in module.parameters():
        p.detach().zero_()
    return module


def conv_nd(dims, *args, **kwargs):
    """Create a 1D, 2D, or 3D convolution module."""
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def normalization(channels, **kwargs):
    """Make a standard normalization layer (GroupNorm with 32 groups)."""
    return nn.GroupNorm(32, channels, **kwargs)


def linear(*args, **kwargs):
    """Create a linear module."""
    return nn.Linear(*args, **kwargs)


def default(value, default_value):
    return value if value is not None else default_value


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Equivalent to torch.repeat_interleave(x, dim=2, repeats=n_rep).
    Input:  (batch, seqlen, num_key_value_heads, head_dim)
    Output: (batch, seqlen, num_attention_heads, head_dim)
    """
    batch, slen, num_key_value_heads, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, :, None, :].expand(
        batch, slen, num_key_value_heads, n_rep, head_dim
    )
    return hidden_states.reshape(batch, slen, num_key_value_heads * n_rep, head_dim)


def real_batched_index_select(t, dim, idx):
    """index_select for batched index and batched t."""
    assert t.ndim >= 2 and idx.ndim >= 2, f"{t.ndim=} {idx.ndim=}"
    assert len(t) == len(idx), f"{len(t)=} != {len(idx)=}"
    return torch.stack(
        [torch.index_select(t[i], dim - 1, idx[i]) for i in range(len(t))]
    )


def _indices_from_mask(mask: torch.Tensor) -> torch.Tensor:
    """Build per-batch token indices without using masked_select on Ascend."""
    mask = mask.bool()
    coords = torch.nonzero(mask, as_tuple=False)
    if coords.numel() == 0:
        return torch.empty(mask.shape[0], 0, device=mask.device, dtype=torch.long)
    return coords[:, 1].reshape(mask.shape[0], -1)


def _gather_tokens_by_index(hidden_states: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    index = index.to(device=hidden_states.device, dtype=torch.long)
    index = index.unsqueeze(-1).expand(-1, -1, hidden_states.shape[-1])
    return hidden_states.gather(dim=1, index=index)


def _scatter_tokens_by_index(
    hidden_states: torch.Tensor, index: torch.Tensor, src: torch.Tensor
) -> torch.Tensor:
    index = index.to(device=hidden_states.device, dtype=torch.long)
    index = index.unsqueeze(-1).expand(-1, -1, hidden_states.shape[-1])
    return hidden_states.scatter(dim=1, index=index, src=src)


def _maybe_forward_context():
    try:
        get_forward_context()
        return nullcontext()
    except AssertionError:
        return set_forward_context(current_timestep=0, attn_metadata=None)


# ---------------------------------------------------------------------------
# RoPE utilities
# ---------------------------------------------------------------------------


def _to_tuple(x, dim=2):
    if isinstance(x, int):
        return (x,) * dim
    elif len(x) == dim:
        return x
    else:
        raise ValueError(f"Expected length {dim} or int, but got {x}")


def get_meshgrid_nd(start, *args, dim=2):
    if len(args) == 0:
        num = _to_tuple(start, dim=dim)
        start = (0,) * dim
        stop = num
    elif len(args) == 1:
        start = _to_tuple(start, dim=dim)
        stop = _to_tuple(args[0], dim=dim)
        num = [stop[i] - start[i] for i in range(dim)]
        num_int = [int(x) for x in num]
        assert (torch.tensor(num) == torch.tensor(num_int)).all(), (
            f"num should be int, but got {num}"
        )
        num = num_int
    elif len(args) == 2:
        start = _to_tuple(start, dim=dim)
        stop = _to_tuple(args[0], dim=dim)
        num = _to_tuple(args[1], dim=dim)
    else:
        raise ValueError(f"len(args) should be 0, 1 or 2, but got {len(args)}")

    axis_grid = []
    for i in range(dim):
        a, b, n = start[i], stop[i], num[i]
        g = torch.linspace(a, b, n + 1, dtype=torch.float32)[:n]
        axis_grid.append(g)
    grid = torch.meshgrid(*axis_grid, indexing="ij")
    grid = torch.stack(grid, dim=0)
    return grid


def build_2d_rope(
    seq_len: int,
    n_elem: int,
    image_infos: list[tuple[slice, tuple[int, int]]] | None = None,
    device: torch.device | None = None,
    base: int = 10000,
    base_rescale_factor: float = 1.0,
    return_all_pos: bool = False,
):
    assert n_elem % 4 == 0, f"n_elem must be divisible by 4, but got {n_elem}."

    if base_rescale_factor != 1.0:
        base *= base_rescale_factor ** (n_elem / (n_elem - 2))
    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, device=device).float() / n_elem))
    theta = theta.reshape(1, n_elem // 4, 2)

    if image_infos is None:
        image_infos = []

    image_infos_list = [image_infos]
    sample_seq_lens = [seq_len]

    x_sections = []
    y_sections = []
    for sample_id, sample_image_infos in enumerate(image_infos_list):
        last_pos = 0
        for sec_slice, (h, w) in sample_image_infos:
            L = sec_slice.start
            if last_pos < L:
                y_sections.append(torch.arange(last_pos, L))
                x_sections.append(torch.arange(last_pos, L))
            elif h is None:
                y_sections.append(torch.arange(sec_slice.start, sec_slice.stop))
                x_sections.append(torch.arange(sec_slice.start, sec_slice.stop))
                continue
            else:
                pass
            beta_y = L + (w * h - h) / 2
            beta_x = L + (w * h - w) / 2
            grid = get_meshgrid_nd((beta_y, beta_x), (beta_y + h, beta_x + w))
            grid = grid.reshape(2, -1)
            y_sections.append(grid[0])
            x_sections.append(grid[1])
            last_pos = L + w * h
        y_sections.append(torch.arange(last_pos, sample_seq_lens[sample_id]))
        x_sections.append(torch.arange(last_pos, sample_seq_lens[sample_id]))

    x_pos = torch.cat(x_sections).long()
    y_pos = torch.cat(y_sections).long()
    x_pos = x_pos[:seq_len]
    y_pos = y_pos[:seq_len]
    all_pos = torch.stack((y_pos, x_pos), dim=1).unsqueeze(1).to(device)

    idx_theta = (all_pos * theta).reshape(all_pos.shape[0], n_elem // 2).repeat(1, 1)

    cos = torch.cos(idx_theta)
    sin = torch.sin(idx_theta)
    cos = torch.cat((cos, cos), dim=-1)
    sin = torch.cat((sin, sin), dim=-1)

    if return_all_pos:
        return cos, sin, all_pos

    return cos, sin


def build_batch_2d_rope(
    seq_len: int,
    n_elem: int,
    image_infos: list[list[tuple[slice, tuple[int, int]]]] | None = None,
    device: torch.device | None = None,
    base: int = 10000,
    base_rescale_factor: float = 1.0,
    return_all_pos: bool = False,
):
    cos_list, sin_list, all_pos_list = [], [], []
    if image_infos is None:
        image_infos = [None]
    for i, image_info in enumerate(image_infos):
        res = build_2d_rope(
            seq_len,
            n_elem,
            image_infos=image_info,
            device=device,
            base=base,
            base_rescale_factor=base_rescale_factor,
            return_all_pos=return_all_pos,
        )
        if isinstance(res, tuple) and len(res) == 3:
            cos, sin, all_pos = res
        elif isinstance(res, tuple) and len(res) == 2:
            cos, sin = res
            all_pos = None
        else:
            raise ValueError(
                "build_2d_rope must return a tuple of length 2 or 3 "
                f"when return_all_pos={return_all_pos}, got: {type(res)} with length "
                f"{len(res) if isinstance(res, tuple) else 'N/A'}"
            )
        cos_list.append(cos)
        sin_list.append(sin)
        all_pos_list.append(all_pos)
    stacked_cos = torch.stack(cos_list, dim=0)
    stacked_sin = torch.stack(sin_list, dim=0)
    if return_all_pos:
        return stacked_cos, stacked_sin, all_pos_list

    return stacked_cos, stacked_sin


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class HunyuanImage3LayerKVCache:
    """Layer-local KV cache for HunyuanImage3 AR text generation."""

    def __init__(self, max_cache_len: int | None = None):
        self.k_cache: torch.Tensor | None = None
        self.v_cache: torch.Tensor | None = None
        self.max_cache_len = max_cache_len
        self.cache_len = 0

    def _target_len(
        self,
        key_states: torch.Tensor,
        cache_position: torch.Tensor | None,
        cache_end: int | None,
    ) -> int:
        if cache_end is not None:
            needed_len = cache_end
        elif cache_position is None:
            needed_len = self.cache_len + key_states.shape[2]
        else:
            needed_len = int(cache_position.max().item()) + 1
        if self.max_cache_len is not None:
            return max(self.max_cache_len, needed_len)
        return needed_len

    def _ensure_cache(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_position: torch.Tensor | None,
        cache_end: int | None,
    ) -> None:
        target_len = self._target_len(key_states, cache_position, cache_end)
        if self.k_cache is not None and self.k_cache.shape[2] >= target_len:
            return

        k_cache = key_states.new_empty(
            key_states.shape[0],
            key_states.shape[1],
            target_len,
            key_states.shape[3],
        )
        v_cache = value_states.new_empty(
            value_states.shape[0],
            value_states.shape[1],
            target_len,
            value_states.shape[3],
        )
        if self.k_cache is not None and self.cache_len > 0:
            k_cache[:, :, : self.cache_len].copy_(
                self.k_cache[:, :, : self.cache_len]
            )
            v_cache[:, :, : self.cache_len].copy_(
                self.v_cache[:, :, : self.cache_len]
            )
        self.k_cache = k_cache
        self.v_cache = v_cache

    def store(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_position: torch.Tensor | None = None,
        cache_end: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if cache_position is not None:
            cache_position = cache_position.to(
                device=key_states.device, dtype=torch.long
            )

        self._ensure_cache(key_states, value_states, cache_position, cache_end)
        assert self.k_cache is not None and self.v_cache is not None

        if cache_position is None:
            start = self.cache_len
            end = start + key_states.shape[2]
            self.k_cache[:, :, start:end].copy_(key_states)
            self.v_cache[:, :, start:end].copy_(value_states)
        elif cache_position.dim() == 1:
            self.k_cache.index_copy_(2, cache_position, key_states)
            self.v_cache.index_copy_(2, cache_position, value_states)
            end = (
                cache_end
                if cache_end is not None
                else int(cache_position[-1].item()) + 1
            )
        elif cache_position.dim() == 2:
            if cache_position.shape[0] != self.k_cache.shape[0]:
                raise ValueError(
                    "cache_position batch size must match cache batch size, got "
                    f"{cache_position.shape[0]} and {self.k_cache.shape[0]}"
                )
            for batch_idx in range(cache_position.shape[0]):
                self.k_cache[batch_idx].index_copy_(
                    1, cache_position[batch_idx], key_states[batch_idx]
                )
                self.v_cache[batch_idx].index_copy_(
                    1, cache_position[batch_idx], value_states[batch_idx]
                )
            end = (
                cache_end
                if cache_end is not None
                else int(cache_position.max().item()) + 1
            )
        else:
            raise ValueError(
                "cache_position must be 1D or 2D, got "
                f"{tuple(cache_position.shape)}"
            )

        self.cache_len = max(self.cache_len, end)
        return self.get()

    def get(self) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.k_cache is not None and self.v_cache is not None
        return (
            self.k_cache[:, :, : self.cache_len],
            self.v_cache[:, :, : self.cache_len],
        )

    def clear(self) -> None:
        self.k_cache = None
        self.v_cache = None
        self.cache_len = 0


class HunyuanImage3ARCache:
    """Container for per-layer HunyuanImage3 AR KV caches."""

    def __init__(self, num_layers: int, max_cache_len: int | None = None):
        self.num_layers = num_layers
        self.caches = [
            HunyuanImage3LayerKVCache(max_cache_len=max_cache_len)
            for _ in range(num_layers)
        ]

    def __getitem__(self, layer_idx: int) -> HunyuanImage3LayerKVCache:
        return self.caches[layer_idx]

    def clear(self) -> None:
        for cache in self.caches:
            cache.clear()


def apply_rotary_pos_emb(
    q, k, cos, sin, position_ids=None, unsqueeze_dim=1, mla=False
) -> tuple[torch.Tensor, torch.Tensor]:
    if position_ids is not None:
        cos = cos[position_ids]
        sin = sin[position_ids]

    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    if mla:
        b, h, s, d = q.shape
        q = q.reshape(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

        b, h, s, d = k.shape
        k = k.reshape(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# ---------------------------------------------------------------------------
# Resolution bucketing
# ---------------------------------------------------------------------------

HUNYUAN_IMAGE3_EXTRA_RESOLUTIONS: tuple[str, ...] = (
    "1024x768",
    "1280x720",
    "768x1024",
    "720x1280",
)


class Resolution:
    def __init__(self, size, *args):
        if isinstance(size, str):
            if "x" in size:
                size = size.split("x")
                size = (int(size[0]), int(size[1]))
            else:
                size = int(size)
        if len(args) > 0:
            size = (size, args[0])
        if isinstance(size, int):
            size = (size, size)

        self.h = self.height = size[0]
        self.w = self.width = size[1]
        self.r = self.ratio = self.height / self.width

    def __getitem__(self, idx):
        if idx == 0:
            return self.h
        elif idx == 1:
            return self.w
        else:
            raise IndexError(f"Index {idx} out of range")

    def __str__(self):
        return f"{self.h}x{self.w}"


class ResolutionGroup:
    def __init__(self, base_size=None, step=None, align=1, extra_resolutions=None):
        self.align = align
        self.base_size = base_size
        assert base_size % align == 0, (
            f"base_size {base_size} is not divisible by align {align}"
        )
        if base_size is not None and not isinstance(base_size, int):
            raise ValueError(f"base_size must be None or int, but got {type(base_size)}")
        if step is None:
            step = base_size // 16
        if step is not None and step > base_size // 2:
            raise ValueError(
                f"step must be smaller than base_size // 2, but got {step} > {base_size // 2}"
            )

        self.step = step
        self.data = self._calc_by_step()

        if extra_resolutions is not None:
            for er in extra_resolutions:
                if not any(r.ratio == er.ratio for r in self.data):
                    self.data.append(er)

        self.ratio = np.array([x.ratio for x in self.data])
        self.attr = ["" for _ in range(len(self.data))]
        self.prefix_space = 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __repr__(self):
        prefix = self.prefix_space * " "
        prefix_close = (self.prefix_space - 4) * " "
        res_str = f"ResolutionGroup(base_size={self.base_size}, step={self.step}, data="
        attr_maxlen = max([len(x) for x in self.attr] + [5])
        res_str += (
            f"\n{prefix}ID: height width   ratio {' ' * max(0, attr_maxlen - 4)}count  h/16 w/16    tokens\n{prefix}"
        )
        res_str += ("\n" + prefix).join(
            [
                f"{i:2d}: ({x.h:4d}, {x.w:4d})  {self.ratio[i]:.4f}  {self.attr[i]:>{attr_maxlen}s}  "
                f"({x.h // 16:3d}, {x.w // 16:3d})  {x.h // 16 * x.w // 16:6d}"
                for i, x in enumerate(self.data)
            ]
        )
        res_str += f"\n{prefix_close})"
        return res_str

    def _calc_by_step(self):
        assert self.align <= self.step, (
            f"align {self.align} must be smaller than step {self.step}"
        )

        min_height = self.base_size // 2
        min_width = self.base_size // 2
        max_height = self.base_size * 2
        max_width = self.base_size * 2

        resolutions = [Resolution(self.base_size, self.base_size)]

        cur_height, cur_width = self.base_size, self.base_size
        while True:
            if cur_height >= max_height and cur_width <= min_width:
                break
            cur_height = min(cur_height + self.step, max_height)
            cur_width = max(cur_width - self.step, min_width)
            resolutions.append(
                Resolution(
                    cur_height // self.align * self.align,
                    cur_width // self.align * self.align,
                )
            )

        cur_height, cur_width = self.base_size, self.base_size
        while True:
            if cur_height <= min_height and cur_width >= max_width:
                break
            cur_height = max(cur_height - self.step, min_height)
            cur_width = min(cur_width + self.step, max_width)
            resolutions.append(
                Resolution(
                    cur_height // self.align * self.align,
                    cur_width // self.align * self.align,
                )
            )

        resolutions = sorted(resolutions, key=lambda x: x.ratio)
        return resolutions

    def get_target_size(self, width, height):
        ratio = height / width
        idx = np.argmin(np.abs(self.ratio - ratio))
        reso = self.data[idx]
        return reso.w, reso.h

    def get_base_size_and_ratio_index(self, width, height):
        ratio = height / width
        idx = np.argmin(np.abs(self.ratio - ratio))
        return self.base_size, idx


# ---------------------------------------------------------------------------
# Timestep embedding helper
# ---------------------------------------------------------------------------


def timestep_embedding(
    t: torch.Tensor, dim: int, max_period: float = 10000.0, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Create sinusoidal timestep embeddings (GLIDE/DiT convention)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=dtype, device=t.device)
        / half
    )
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat(
            [embedding, torch.zeros_like(embedding[:, :1], dtype=dtype, device=t.device)], dim=-1
        )
    return embedding


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""

    def __init__(
        self,
        hidden_size,
        act_layer=nn.GELU,
        frequency_embedding_size=256,
        max_period=10000,
        out_size=None,
        dtype=None,
        device=None,
    ):
        factory_kwargs = {"dtype": dtype, "device": device}
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period
        if out_size is None:
            out_size = hidden_size

        self.mlp = nn.Sequential(
            nn.Linear(
                frequency_embedding_size, hidden_size, bias=True, **factory_kwargs
            ),
            act_layer(),
            nn.Linear(hidden_size, out_size, bias=True, **factory_kwargs),
        )
        nn.init.normal_(self.mlp[0].weight, std=0.02)
        nn.init.normal_(self.mlp[2].weight, std=0.02)

    def forward(self, t):
        t_freq = timestep_embedding(t, self.frequency_embedding_size, self.max_period).type(
            self.mlp[0].weight.dtype
        )
        t_emb = self.mlp(t_freq)
        return t_emb


class ResBlock(nn.Module):
    """A residual block that can optionally change the number of channels."""

    def __init__(
        self,
        in_channels,
        emb_channels,
        out_channels=None,
        dropout=0.0,
        use_conv=False,
        dims=2,
        up=False,
        down=False,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"dtype": dtype, "device": device}
        super().__init__()
        self.in_channels = in_channels
        self.dropout = dropout
        self.out_channels = out_channels or self.in_channels
        self.use_conv = use_conv

        self.in_layers = nn.Sequential(
            normalization(self.in_channels, **factory_kwargs),
            nn.SiLU(),
            conv_nd(
                dims, self.in_channels, self.out_channels, 3, padding=1, **factory_kwargs
            ),
        )

        self.updown = up or down
        self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(), linear(emb_channels, 2 * self.out_channels, **factory_kwargs)
        )

        self.out_layers = nn.Sequential(
            normalization(self.out_channels, **factory_kwargs),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(
                    dims,
                    self.out_channels,
                    self.out_channels,
                    3,
                    padding=1,
                    **factory_kwargs,
                )
            ),
        )

        if self.out_channels == self.in_channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, self.in_channels, self.out_channels, 3, padding=1, **factory_kwargs
            )
        else:
            self.skip_connection = conv_nd(
                dims, self.in_channels, self.out_channels, 1, **factory_kwargs
            )

    def forward(self, x, emb) -> torch.Tensor:
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        emb_out = self.emb_layers(emb)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]

        out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
        scale, shift = torch.chunk(emb_out, 2, dim=1)
        h = out_norm(h) * (1.0 + scale) + shift
        h = out_rest(h)

        return self.skip_connection(x) + h


class UNetDown(nn.Module):
    """Patch-embed: noise latents [B, C, H, W] -> token sequence [B, H'*W', D]."""

    def __init__(
        self,
        patch_size,
        in_channels,
        emb_channels,
        hidden_channels,
        out_channels,
        dropout=0.0,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"dtype": dtype, "device": device}
        super().__init__()

        self.patch_size = patch_size
        assert self.patch_size in [1, 2, 4, 8]

        self.model = nn.ModuleList(
            [
                conv_nd(
                    2,
                    in_channels=in_channels,
                    out_channels=hidden_channels,
                    kernel_size=3,
                    padding=1,
                    **factory_kwargs,
                )
            ]
        )

        if self.patch_size == 1:
            self.model.append(
                ResBlock(
                    in_channels=hidden_channels,
                    emb_channels=emb_channels,
                    out_channels=out_channels,
                    dropout=dropout,
                    **factory_kwargs,
                )
            )
        else:
            for i in range(self.patch_size // 2):
                self.model.append(
                    ResBlock(
                        in_channels=hidden_channels,
                        emb_channels=emb_channels,
                        out_channels=(
                            hidden_channels if (i + 1) * 2 != self.patch_size else out_channels
                        ),
                        dropout=dropout,
                        down=True,
                        **factory_kwargs,
                    )
                )

    def forward(self, x, t):
        assert x.shape[2] % self.patch_size == 0 and x.shape[3] % self.patch_size == 0
        for module in self.model:
            if isinstance(module, ResBlock):
                x = module(x, t)
            else:
                x = module(x)
        _, _, token_h, token_w = x.shape
        x = rearrange(x, "b c h w -> b (h w) c")
        return x, token_h, token_w


class UNetUp(nn.Module):
    """Final layer: token sequence [B, H'*W', D] -> noise pred [B, C, H, W]."""

    def __init__(
        self,
        patch_size,
        in_channels,
        emb_channels,
        hidden_channels,
        out_channels,
        dropout=0.0,
        device=None,
        dtype=None,
        out_norm=False,
    ):
        factory_kwargs = {"dtype": dtype, "device": device}
        super().__init__()

        self.patch_size = patch_size
        assert self.patch_size in [1, 2, 4, 8]

        self.model = nn.ModuleList()

        if self.patch_size == 1:
            self.model.append(
                ResBlock(
                    in_channels=in_channels,
                    emb_channels=emb_channels,
                    out_channels=hidden_channels,
                    dropout=dropout,
                    **factory_kwargs,
                )
            )
        else:
            for i in range(self.patch_size // 2):
                self.model.append(
                    ResBlock(
                        in_channels=in_channels if i == 0 else hidden_channels,
                        emb_channels=emb_channels,
                        out_channels=hidden_channels,
                        dropout=dropout,
                        up=True,
                        **factory_kwargs,
                    )
                )

        if out_norm:
            self.model.append(
                nn.Sequential(
                    normalization(hidden_channels, **factory_kwargs),
                    nn.SiLU(),
                    conv_nd(
                        2,
                        in_channels=hidden_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        padding=1,
                        **factory_kwargs,
                    ),
                )
            )
        else:
            self.model.append(
                conv_nd(
                    2,
                    in_channels=hidden_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1,
                    **factory_kwargs,
                )
            )

    def forward(self, x, t, token_h, token_w):
        x = rearrange(x, "b (h w) c -> b c h w", h=token_h, w=token_w)
        for module in self.model:
            if isinstance(module, ResBlock):
                x = module(x, t)
            else:
                x = module(x)
        return x


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


class HunYuanAttention(nn.Module):
    """
    Self-attention for HunyuanImage3 decoder layers.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        use_qk_norm: bool = False,
        rms_norm_eps: float = 1e-5,
        bias: bool = False,
        layer_idx: int = 0,
        supported_attention_backends: set[AttentionBackendEnum] | None = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.total_num_heads = num_heads
        self.total_num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.layer_idx = layer_idx
        self.scaling = head_dim**-0.5
        self.use_qk_norm = use_qk_norm

        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=head_dim,
            total_num_heads=num_heads,
            total_num_kv_heads=num_kv_heads,
            bias=bias,
            prefix=f"layers.{layer_idx}.self_attn.qkv_proj",
        )
        self.num_heads = self.qkv_proj.num_heads
        self.num_kv_heads = self.qkv_proj.num_kv_heads
        self.q_size = self.num_heads * head_dim
        self.kv_size = self.num_kv_heads * head_dim

        self.o_proj = RowParallelLinear(
            input_size=num_heads * head_dim,
            output_size=hidden_size,
            bias=bias,
            input_is_parallel=True,
            prefix=f"layers.{layer_idx}.self_attn.o_proj",
        )
        self.attn = USPAttention(
            num_heads=self.num_heads,
            head_size=head_dim,
            num_kv_heads=self.num_kv_heads,
            softmax_scale=self.scaling,
            causal=False,
            skip_sequence_parallel=False,
            supported_attention_backends=supported_attention_backends,
        )

        if self.use_qk_norm:
            self.query_layernorm = RMSNorm(head_dim, eps=rms_norm_eps)
            self.key_layernorm = RMSNorm(head_dim, eps=rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rope_2d: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        kv_cache: HunyuanImage3LayerKVCache | None = None,
        cache_position: torch.Tensor | None = None,
        cache_end: int | None = None,
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()

        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.reshape(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply 2D RoPE if provided
        if rope_2d is not None:
            cos, sin = rope_2d
            # cos/sin: [B, L, D_h] -> [B, 1, L, D_h]
            cos = cos.unsqueeze(1).to(device=q.device, dtype=q.dtype)
            sin = sin.unsqueeze(1).to(device=q.device, dtype=q.dtype)
            q = (q * cos) + (rotate_half(q) * sin)
            k = (k * cos) + (rotate_half(k) * sin)

        if self.use_qk_norm:
            q = self.query_layernorm(q)
            k = self.key_layernorm(k)

        q = q.to(v.dtype)
        k = k.to(v.dtype)

        if kv_cache is not None:
            k, v = kv_cache.store(k, v, cache_position, cache_end)
            q = q.to(k.dtype)

        # GQA expansion is an architectural property, not mask-dependent.
        # Must happen unconditionally so that q/k/v head dimensions match
        # what USPAttention expects (including AR decode steps where
        # attention_mask is None).
        if self.num_kv_heads < self.num_heads:
            n_rep = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)

        q_backend = q.transpose(1, 2)  # [B, S, H, D_h]
        k_backend = k.transpose(1, 2)
        v_backend = v.transpose(1, 2)

        # Sequence Parallelism + 4D causal mask handling:
        # USPAttention's SP path only supports 2D [B, S_local] masks.
        # When SP is enabled with a 4D mask, we gather KV across SP ranks
        # and skip SP for this attention call, following the LTX-2 pattern.
        sp_size = get_sequence_parallel_world_size()
        skip_sp_override = False
        if attention_mask is not None and sp_size > 1:
            # Gather k/v across SP ranks so each rank sees the full sequence
            k_backend = sequence_model_parallel_all_gather(
                k_backend.contiguous(), dim=1
            )
            v_backend = sequence_model_parallel_all_gather(
                v_backend.contiguous(), dim=1
            )
            # The mask is also sharded by SP; gather to match the full KV length
            if attention_mask.dim() == 4:
                # 4D mask [B, 1, seq_len, seq_len] -> gather the last dim
                gathered_mask = sequence_model_parallel_all_gather(
                    attention_mask, dim=3
                )
                attention_mask = gathered_mask
            elif attention_mask.dim() == 2:
                # 2D mask [B, seq_len] -> gather dim 1
                gathered_mask = sequence_model_parallel_all_gather(
                    attention_mask, dim=1
                )
                attention_mask = gathered_mask
            skip_sp_override = True

        with _maybe_forward_context():
            attn_output = self.attn(
                q_backend,
                k_backend,
                v_backend,
                attn_mask=attention_mask,
                skip_sequence_parallel_override=skip_sp_override,
            )

        # [B, L, H_local, D_h] -> [B, L, H_local*D_h]
        attn_output = attn_output.contiguous().view(bsz, q_len, -1)
        output, _ = self.o_proj(attn_output)
        return output


# ---------------------------------------------------------------------------
# MLP / MoE
# ---------------------------------------------------------------------------


class HunYuanMLP(nn.Module):
    """Feed-forward block with SiLU-gated projection (gate + up packed)."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        bias: bool = False,
        prefix: str = "",
    ):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size, intermediate_size],
            bias=bias,
            gather_output=False,
            prefix=f"{prefix}.gate_up_proj" if prefix else "gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=bias,
            input_is_parallel=True,
            prefix=f"{prefix}.down_proj" if prefix else "down_proj",
        )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class HunYuanSparseMoeBlock(nn.Module):
    """Sparse MoE block backed by SRT's FusedMoE + TopK.

    Delegates all expert computation (routing, dispatch, fused kernel
    execution, combine) to SRT's high-performance MoE stack, which
    auto-selects the best kernel per platform (Triton on GPU, ascend_fuseep
    or grouped-matmul on NPU).  Expert Parallelism is fully supported via
    the bridge initialised in ``srt_moe_bridge``.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int = 1,
        num_shared_expert: int = 0,
        bias: bool = False,
        prefix: str = "",
        alt_stream: torch.cuda.Stream | None = None,
        layer_id: int = 0,
        quant_config=None,
    ):
        super().__init__()
        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
        from sglang.srt.layers.moe.topk import TopK

        # Lazily initialize SRT MoE config on first MoE layer creation.
        # Parallel groups are already set up at launch time; this only
        # sets MoE-specific globals (MOE_A2A_BACKEND, etc.).
        from sglang.multimodal_gen.runtime.distributed.srt_moe_bridge import (
            init_srt_moe_config,
        )
        init_srt_moe_config()

        self.alt_stream = alt_stream

        # FP32 router gate (replicated on all ranks)
        self.gate = ReplicatedLinear(
            hidden_size,
            num_experts,
            bias=False,
            params_dtype=torch.float32,
            prefix=f"{prefix}.gate" if prefix else "gate",
        )

        # SRT fused TopK (GPU: sgl_kernel.topk_softmax; NPU: npu_moe_gating_top_k_softmax)
        self.topk = TopK(top_k=top_k, renormalize=True, scoring_func="softmax")

        # NOTE: ``reduce_results`` must be ``False`` for DeepEP (and any
        # other all-to-all dispatcher such as Mooncake / NIXL / Mori).
        # These dispatchers handle cross-rank token routing entirely
        # inside their own ``dispatch`` / ``combine`` phases — the
        # combine step already returns the complete weighted sum of
        # expert outputs for every input token on each rank.  An extra
        # ``all_reduce`` would multiply the result by ``ep_size`` and
        # corrupt the output.  For StandardDispatcher + TP the
        # all-reduce is still needed and is handled inside FusedMoE via
        # ``reduce_results=True``.
        from sglang.srt.layers.moe.utils import get_moe_a2a_backend

        a2a_backend = get_moe_a2a_backend()
        self._use_deepep_or_similar = (
            a2a_backend.is_deepep()
            or a2a_backend.is_mooncake()
            or a2a_backend.is_nixl()
            or a2a_backend.is_mori()
        )

        # SRT FusedMoE — owns w13/w2 weights, dispatch, fused kernels, combine
        self.experts = FusedMoE(
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            layer_id=layer_id,
            top_k=top_k,
            reduce_results=not self._use_deepep_or_similar,
            quant_config=quant_config,
            prefix=f"{prefix}.experts" if prefix else "experts",
        )

        # Shared expert
        self.shared_mlp = None
        if num_shared_expert > 0:
            self.shared_mlp = HunYuanMLP(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size * num_shared_expert,
                bias=bias,
                prefix=f"{prefix}.shared_mlp" if prefix else "shared_mlp",
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        hidden_dim = hidden_states.shape[-1]
        hidden_states_flat = hidden_states.view(-1, hidden_dim)

        # FP32 router
        router_logits, _ = self.gate(hidden_states_flat.float())
        topk_output = self.topk(hidden_states_flat, router_logits)

        if self.alt_stream is not None and self.shared_mlp is not None:
            # Dual-stream: shared_mlp on main stream, experts on alt stream
            current_stream = torch.cuda.current_stream()
            shared_output = self.shared_mlp(hidden_states)

            self.alt_stream.wait_stream(current_stream)
            with torch.cuda.stream(self.alt_stream):
                routed_output = self.experts(
                    hidden_states_flat, topk_output
                ).view(orig_shape)
            current_stream.wait_stream(self.alt_stream)

            output = routed_output + shared_output
        else:
            # Single-stream: sequential execution
            routed_output = self.experts(
                hidden_states_flat, topk_output
            ).view(orig_shape)
            output = routed_output
            if self.shared_mlp is not None:
                output = output + self.shared_mlp(hidden_states)

        return output


# ---------------------------------------------------------------------------
# Decoder layer
# ---------------------------------------------------------------------------


class HunyuanImage3DecoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        use_qk_norm: bool = False,
        rms_norm_eps: float = 1e-5,
        attention_bias: bool = False,
        num_experts: int = 0,
        moe_topk: int = 1,
        num_shared_expert: int = 0,
        moe_intermediate_size: int | None = None,
        layer_idx: int = 0,
        moe_layer_num_skipped: int = 0,
        supported_attention_backends: set[AttentionBackendEnum] | None = None,
        alt_stream: torch.cuda.Stream | None = None,
        quant_config=None,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        self.self_attn = HunYuanAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            use_qk_norm=use_qk_norm,
            rms_norm_eps=rms_norm_eps,
            bias=attention_bias,
            layer_idx=layer_idx,
            supported_attention_backends=supported_attention_backends,
        )

        use_moe = num_experts > 1 and layer_idx >= moe_layer_num_skipped
        if use_moe:
            moe_inter = moe_intermediate_size or intermediate_size
            self.mlp = HunYuanSparseMoeBlock(
                hidden_size=hidden_size,
                intermediate_size=moe_inter,
                num_experts=num_experts,
                top_k=moe_topk,
                num_shared_expert=num_shared_expert,
                prefix=f"layers.{layer_idx}.mlp",
                alt_stream=alt_stream,
                layer_id=layer_idx,
                quant_config=quant_config,
            )
        else:
            # Dense MLP
            self.mlp = HunYuanMLP(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                prefix=f"layers.{layer_idx}.mlp",
            )

        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rope_2d: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        kv_cache: HunyuanImage3LayerKVCache | None = None,
        cache_position: torch.Tensor | None = None,
        cache_end: int | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            rope_2d=rope_2d,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
            cache_position=cache_position,
            cache_end=cache_end,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


# ---------------------------------------------------------------------------
# Main DiT model
# ---------------------------------------------------------------------------


class HunyuanImage3DiT(CachableDiT, LayerwiseOffloadableModuleMixin):
    """
    HunyuanImage-3.0 Diffusion Transformer backbone.

    This model implements the DiT architecture for HunyuanImage-3.0, adapted to
    sglang's CachableDiT base class. It handles the forward pass for a single
    denoising step: embedding noise latents + text tokens, applying 2D RoPE,
    running through transformer layers, and projecting back to latent space.

    Pipeline-level logic (scheduler, CFG, VAE decode, etc.) is handled by the
    sglang pipeline framework, NOT by this class.
    """

    _aliases = ["HunyuanImage3ForCausalMM"]

    _compile_conditions = HunyuanImage3DiTConfig().arch_config._compile_conditions
    param_names_mapping = HunyuanImage3DiTConfig().arch_config.param_names_mapping
    reverse_param_names_mapping = HunyuanImage3DiTConfig().arch_config.reverse_param_names_mapping
    lora_param_names_mapping = HunyuanImage3DiTConfig().arch_config.lora_param_names_mapping

    _qkv_checkpoint_suffixes = (
        ".self_attn.qkv_proj.weight",
        ".self_attn.qkv_proj.weight_scale",
    )

    def _split_checkpoint_qkv_weight(self, qkv: torch.Tensor) -> torch.Tensor:
        """Convert HunyuanImage3 grouped QKV checkpoint layout to packed QKV."""
        if qkv.dim() == 0:
            return qkv

        num_kv_groups = self.num_attention_heads // self.num_kv_heads
        expected_size = self.num_kv_heads * (num_kv_groups + 2) * self.head_dim
        if qkv.shape[0] != expected_size:
            return qkv

        trailing_shape = qkv.shape[1:]
        qkv = qkv.reshape(
            self.num_kv_heads,
            num_kv_groups + 2,
            self.head_dim,
            *trailing_shape,
        )
        q, k, v = torch.split(qkv, [num_kv_groups, 1, 1], dim=1)
        return torch.cat(
            (
                q.reshape(-1, *trailing_shape),
                k.reshape(-1, *trailing_shape),
                v.reshape(-1, *trailing_shape),
            ),
            dim=0,
        )

    @staticmethod
    def _swap_checkpoint_gate_up(gate_up: torch.Tensor) -> torch.Tensor:
        """Convert checkpoint [up, gate] layout to SiluAndMul [gate, up]."""
        if gate_up.dim() == 0 or gate_up.shape[0] % 2 != 0:
            return gate_up
        up, gate = gate_up.chunk(2, dim=0)
        return torch.cat((gate, up), dim=0)

    def convert_checkpoint_weight_for_loading(
        self,
        *,
        source_name: str,
        target_name: str,
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        if target_name.endswith(self._qkv_checkpoint_suffixes):
            return self._split_checkpoint_qkv_weight(tensor)
        if ".gate_and_up_proj." in source_name and ".gate_up_proj." in target_name:
            return self._swap_checkpoint_gate_up(tensor)
        return tensor

    def __init__(
        self,
        config: HunyuanImage3DiTConfig,
        hf_config: dict[str, Any],
        quant_config=None,
    ):
        super().__init__(config=config, hf_config=hf_config)
        self.alt_stream = torch.cuda.Stream() if is_cuda() else None

        arch = config.arch_config
        self.patch_size = arch.patch_size
        self.in_channels = arch.in_channels
        self.out_channels = arch.out_channels
        self.vocab_size = arch.vocab_size
        self.image_base_size = arch.image_base_size
        self.vae_downsample_factor = arch.vae_downsample_factor
        self.rope_theta = arch.rope_theta
        self.rope_axes_dim = arch.rope_axes_dim

        # Instance attributes required by BaseDiT
        self.hidden_size = arch.hidden_size
        self.num_attention_heads = arch.num_attention_heads
        self.num_channels_latents = arch.in_channels
        self._supported_attention_backends = arch._supported_attention_backends

        head_dim = arch.attention_head_dim
        num_kv_heads = arch.num_key_value_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        # Sequence parallelism
        self.sp_size = get_sp_world_size()

        # Token embedding / LM head use vocab parallelism to match the TP
        # execution style used by SGLang language models.
        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            arch.hidden_size,
            org_num_embeddings=self.vocab_size,
            prefix="embed_tokens",
        )

        # Per-layer MoE parameters: config.json uses lists for moe_topk,
        # moe_intermediate_size, num_shared_expert — one entry per layer.
        _maybe_list = lambda v, idx: v[idx] if isinstance(v, (list, tuple)) else v

        # Transformer layers
        self.layers = nn.ModuleList(
            [
                HunyuanImage3DecoderLayer(
                    hidden_size=arch.hidden_size,
                    intermediate_size=arch.intermediate_size
                    or int(arch.hidden_size * arch.mlp_ratio),
                    num_heads=arch.num_attention_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    use_qk_norm=arch.qk_norm == "rms_norm",
                    rms_norm_eps=arch.rms_norm_eps,
                    attention_bias=arch.attention_bias,
                    num_experts=_maybe_list(arch.num_experts, i),
                    moe_topk=_maybe_list(arch.moe_topk, i),
                    num_shared_expert=_maybe_list(arch.num_shared_expert, i),
                    moe_intermediate_size=_maybe_list(arch.moe_intermediate_size, i),
                    layer_idx=i,
                    moe_layer_num_skipped=arch.moe_layer_num_skipped,
                    supported_attention_backends=self._supported_attention_backends,
                    alt_stream=self.alt_stream,
                    quant_config=quant_config,
                )
                for i in range(arch.num_hidden_layers)
            ]
        )
        self.layer_names = ["layers"]

        # Final norm
        self.norm = RMSNorm(arch.hidden_size, eps=arch.rms_norm_eps)

        # AR text generation head (independent weights, not tied to embed_tokens)
        self.lm_head = VocabParallelEmbedding(
            self.vocab_size,
            arch.hidden_size,
            org_num_embeddings=self.vocab_size,
            prefix="lm_head",
        )
        if arch.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        self._lm_head_mapping: torch.Tensor | None = None

        # Timestep embedders (local implementation; checkpoint uses mlp.0/mlp.2 naming)
        self.time_embed = TimestepEmbedder(hidden_size=arch.hidden_size)
        self.timestep_emb = TimestepEmbedder(hidden_size=arch.hidden_size)
        self.time_embed_2 = TimestepEmbedder(hidden_size=arch.hidden_size)

        # Patch embed (noise latents -> hidden_size tokens)
        vae_latent_channels = arch.in_channels
        self.patch_embed = UNetDown(
            patch_size=arch.patch_size,
            emb_channels=arch.hidden_size,
            in_channels=vae_latent_channels,
            hidden_channels=arch.patch_embed_hidden_dim,
            out_channels=arch.hidden_size,
        )

        # Final layer (hidden_size tokens -> noise pred latents)
        self.final_layer = UNetUp(
            patch_size=arch.patch_size,
            emb_channels=arch.hidden_size,
            in_channels=arch.hidden_size,
            hidden_channels=arch.patch_embed_hidden_dim,
            out_channels=vae_latent_channels,
            out_norm=True,
        )

        # Resolution group for bucketing
        self.reso_group = ResolutionGroup(
            base_size=arch.image_base_size,
            extra_resolutions=[
                Resolution(s) for s in HUNYUAN_IMAGE3_EXTRA_RESOLUTIONS
            ],
        )

        self._routed_expert_pattern = re.compile(
            r"^(?P<prefix>.*?\.mlp)\.experts\.(?P<expert_id>\d+)\.(?P<proj>gate_up_proj|down_proj)\.weight$"
        )
        self._teacache_skipped_steps = 0

    def _load_routed_expert_weight(
        self,
        params_dict: dict[str, nn.Parameter],
        name: str,
        loaded_weight: torch.Tensor,
    ) -> str | None:
        """Load routed expert weights via SRT FusedMoE's weight_loader.

        Checkpoint naming: ``layers.X.mlp.experts.{i}.gate_up_proj.weight``
        and ``layers.X.mlp.experts.{i}.down_proj.weight`` are remapped to
        FusedMoE's ``w13_weight`` / ``w2_weight`` using shard IDs
        ``w1``/``w3`` (gate/up halves) / ``w2``.
        """
        routed_match = self._routed_expert_pattern.match(name)
        if routed_match is None:
            return None

        prefix = routed_match.group("prefix")
        expert_id = int(routed_match.group("expert_id"))
        proj = routed_match.group("proj")

        if proj == "gate_up_proj":
            target_name = f"{prefix}.experts.w13_weight"
            # gate_up_proj is [2*intermediate, hidden]; split into gate (w1)
            # and up (w3) halves for FusedMoE's weight_loader which accepts
            # shard_id in ("w1", "w2", "w3").
            gate_weight, up_weight = loaded_weight.chunk(2, dim=0)
            for shard_id, weight in [("w1", gate_weight), ("w3", up_weight)]:
                param = params_dict.get(target_name)
                if param is None:
                    continue
                weight_loader = getattr(param, "weight_loader", None)
                if weight_loader is not None:
                    weight_loader(param, weight, target_name, shard_id, expert_id)
            return target_name
        else:
            target_name = f"{prefix}.experts.w2_weight"
            shard_id = "w2"

        param = params_dict.get(target_name)
        if param is None:
            return target_name
        weight_loader = getattr(param, "weight_loader", None)
        if weight_loader is not None:
            weight_loader(param, loaded_weight, target_name, shard_id, expert_id)
        return target_name

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        """Load weights from a checkpoint.

        Routed expert weights are remapped from per-expert checkpoint naming
        (``experts.{i}.gate_up_proj`` / ``experts.{i}.down_proj``) into
        SRT FusedMoE's fused ``w13_weight`` / ``w2_weight`` tensors using
        FusedMoE's built-in weight_loader.
        """
        from sglang.multimodal_gen.runtime.loader.weight_utils import (
            default_weight_loader,
        )

        stacked_params_mapping = [
            # (param_name, weight_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]

        params_dict = dict(self.named_parameters())

        loaded_count = 0
        loaded_params = set()

        for name, loaded_weight in weights:
            # Map "mlp.gate.wg" → "mlp.gate" (HF checkpoint naming)
            if ".mlp.gate.wg." in name:
                name = name.replace(".mlp.gate.wg.", ".mlp.gate.")

            routed_target = self._load_routed_expert_weight(
                params_dict, name, loaded_weight
            )
            if routed_target is not None:
                if routed_target not in params_dict:
                    continue
                loaded_count += 1
                loaded_params.add(routed_target)
                continue

            # Handle stacked params (qkv_proj).  Expert and non-expert
            # gate_up_proj weights are handled natively by
            # MergedColumnParallelLinear's weight_loader.
            is_stacked = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                src = f".{weight_name}."
                dst = f".{param_name}."
                if src not in name:
                    continue
                name = name.replace(src, dst)
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                loaded_count += 1
                loaded_params.add(name)
                is_stacked = True
                break
            if is_stacked:
                continue

            # Fallback: direct load
            if name not in params_dict:
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_count += 1
            loaded_params.add(name)

    def post_load_weights(self) -> None:
        """Post-loading weight processing.

        On NPU, MoE weight processing (transpose + NZ format conversion) is
        deferred to the first forward call via one-shot pre-hooks. By that time
        ``ResidentStrategy`` has already moved the model to NPU, so the
        ``process_weights_after_loading`` call happens in-place without CPU
        round-trips.

        Pattern from vllm-omni ``HunyuanFusedMoEDefault._initialize_kernel_hook``.
        NPU-specific transpose + NZ follows vllm-ascend
        ``AscendUnquantizedFusedMoEMethod.process_weights_after_loading``.
        """
        super().post_load_weights()
        if not is_npu():
            return

        for layer in self.layers:
            mlp = getattr(layer, "mlp", None)
            experts = getattr(mlp, "experts", None) if mlp is not None else None
            if experts is None:
                continue
            # Register a one-shot hook. The hook removes itself after processing,
            # preventing double execution.
            handle = experts.register_forward_pre_hook(self._npu_moe_init_hook)
            experts._npu_moe_init_handle = handle

        logger.info(
            "[NPU] Registered lazy-init hooks for MoE weight processing "
            "(transpose + NZ on first forward)."
        )

    @staticmethod
    def _npu_moe_init_hook(module, args):
        """One-shot hook: transpose + NZ format conversion for MoE weights.

        Called on first forward when model is resident on NPU. Calls
        ``process_weights_after_loading`` which performs:
        1. Weight transpose: [E, N, K] -> [E, K, N] (for npu_grouped_matmul)
        2. NZ format cast: optimal for NPU Cube engine

        After processing, removes itself to prevent double execution.
        Pattern matches vllm-omni ``HunyuanFusedMoEDefault._initialize_kernel_hook``.
        """
        quant_method = getattr(module, "quant_method", None)
        if (
            quant_method is not None
            and hasattr(quant_method, "process_weights_after_loading")
        ):
            quant_method.process_weights_after_loading(module)

        # One-shot: remove after first execution.
        handle = getattr(module, "_npu_moe_init_handle", None)
        if handle is not None:
            handle.remove()

    def _compute_ar_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.lm_head.quant_method.apply(self.lm_head, hidden_states)
        if self.lm_head.tp_size <= 1:
            return logits[..., : self.vocab_size]

        logits = tensor_model_parallel_all_gather(
            logits, dim=-1, tp_group=self.lm_head.tp_group
        )
        if (
            self._lm_head_mapping is None
            or self._lm_head_mapping.device != logits.device
        ):
            mapping = self.lm_head.get_sharded_to_full_mapping()
            if mapping is not None:
                self._lm_head_mapping = torch.tensor(
                    mapping, device=logits.device, dtype=torch.long
                )
        if self._lm_head_mapping is not None:
            logits = logits.index_select(-1, self._lm_head_mapping)
        return logits[..., : self.vocab_size]

    def _forward_ar_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        rope_2d: tuple[torch.Tensor, torch.Tensor] | None = None,
        kv_caches: HunyuanImage3ARCache | None = None,
        cache_position: torch.Tensor | None = None,
        cache_end: int | None = None,
        use_cache: bool = False,
        cond_vae_images: torch.Tensor | None = None,
        cond_timestep: torch.Tensor | None = None,
        cond_vae_scatter_index: torch.Tensor | None = None,
        cond_vit_embeds: torch.Tensor | None = None,
        cond_vit_scatter_index: torch.Tensor | None = None,
        cond_timestep_scatter_index: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, HunyuanImage3ARCache]:
        """Forward text tokens through the root module for AR logits.

        When cond_* tensors are provided (ti2i AR prefill), vision features
        are scattered into the image-token positions right after
        ``embed_tokens``, mirroring the DiT forward's cond injection (steps
        5-6).  Decode steps (step > 0) do not pass cond kwargs, so this
        path is a no-op for single-token decode.
        """
        hidden_states = self.embed_tokens(input_ids)
        bsz = hidden_states.shape[0]

        # -- Inject cond vision features (ti2i AR prefill only) --
        # Mirrors DiT forward steps 5-6 (lines 1909-1945).
        if cond_vae_images is not None and cond_vae_scatter_index is not None:
            cond_ts = cond_timestep if cond_timestep is not None else torch.zeros(
                bsz, device=input_ids.device
            )
            cond_t_emb = self.time_embed(cond_ts)
            cond_patch, _, _ = self.patch_embed(
                cond_vae_images.to(dtype=hidden_states.dtype), cond_t_emb
            )
            hidden_states = _scatter_tokens_by_index(
                hidden_states, cond_vae_scatter_index, cond_patch
            )
            if cond_timestep_scatter_index is not None:
                n_embd = hidden_states.shape[-1]
                cond_ts_emb = self.timestep_emb(cond_ts).reshape(bsz, 1, n_embd)
                hidden_states = hidden_states.scatter(
                    dim=1,
                    index=cond_timestep_scatter_index.unsqueeze(-1).expand(
                        -1, -1, n_embd
                    ),
                    src=cond_ts_emb,
                )
        if cond_vit_embeds is not None and cond_vit_scatter_index is not None:
            bsz = hidden_states.shape[0]
            cond_vit_flat = cond_vit_embeds.reshape(
                bsz, -1, hidden_states.shape[-1]
            ).to(dtype=hidden_states.dtype, device=hidden_states.device)
            hidden_states = hidden_states.scatter(
                dim=1,
                index=cond_vit_scatter_index.to(
                    device=hidden_states.device, dtype=torch.long
                )
                .unsqueeze(-1)
                .expand(-1, -1, hidden_states.shape[-1]),
                src=cond_vit_flat,
            )

        caches = kv_caches if use_cache else None
        for layer_idx, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states=hidden_states,
                rope_2d=rope_2d,
                attention_mask=attention_mask,
                kv_cache=caches[layer_idx] if caches is not None else None,
                cache_position=cache_position,
                cache_end=cache_end,
            )
        hidden_states = self.norm(hidden_states)
        logits = self._compute_ar_logits(hidden_states[:, -1, :])
        if use_cache:
            return logits, caches
        return logits

    def forward(
        self,
        hidden_states: torch.Tensor | None = None,
        timestep: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        image_mask: torch.Tensor | None = None,
        image_scatter_index: torch.Tensor | None = None,
        timestep_scatter_index: torch.Tensor | None = None,
        cond_vae_image_mask: torch.Tensor | None = None,
        cond_vae_scatter_index: torch.Tensor | None = None,
        cond_vit_image_mask: torch.Tensor | None = None,
        cond_vit_scatter_index: torch.Tensor | None = None,
        cond_vit_embeds: torch.Tensor | None = None,
        rope_2d: tuple[torch.Tensor, torch.Tensor] | None = None,
        position_ids: torch.Tensor | None = None,
        token_h: int | None = None,
        token_w: int | None = None,
        input_ids: torch.Tensor | None = None,
        mode: str | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass for a single denoising step.

        Args:
            hidden_states: Noise latents [B, C, H, W].
            timestep: Timestep tensor [B].
            encoder_hidden_states: Full token sequence embeddings [B, seq_len, dim].
            encoder_attention_mask: Attention mask [B, seq_len] or [B, 1, seq_len, seq_len].
            image_mask: Boolean mask [B, seq_len] indicating image token positions.
            image_scatter_index: Precomputed image token positions [B, num_patches].
            timestep_scatter_index: Index tensor for scattering timestep embeddings.
            cond_vae_image_mask: Mask for conditional VAE image positions (I2I).
            cond_vae_scatter_index: Conditional VAE image token positions.
            cond_vit_image_mask: Mask for conditional ViT image positions (I2I).
            cond_vit_scatter_index: Conditional ViT image token positions.
            cond_vit_embeds: ViT condition embeddings for I2I.
            rope_2d: Tuple of (cos, sin) for 2D rotary position embeddings.
            position_ids: Position IDs for the token sequence.
            token_h: Token height for the image.
            token_w: Token width for the image.

        Returns:
            Noise prediction [B, C, H, W].
        """
        if mode == "ar_text":
            if input_ids is None:
                raise ValueError("input_ids must be provided for mode='ar_text'")
            return self._forward_ar_text(
                input_ids=input_ids,
                attention_mask=encoder_attention_mask,
                rope_2d=rope_2d,
                kv_caches=kwargs.get("kv_caches"),
                cache_position=kwargs.get("cache_position"),
                cache_end=kwargs.get("cache_end"),
                use_cache=kwargs.get("use_cache", False),
                cond_vae_images=kwargs.get("cond_vae_images"),
                cond_timestep=kwargs.get("cond_timestep"),
                cond_vae_scatter_index=cond_vae_scatter_index,
                cond_vit_embeds=cond_vit_embeds,
                cond_vit_scatter_index=cond_vit_scatter_index,
                cond_timestep_scatter_index=kwargs.get("cond_timestep_scatter_index"),
            )

        if hidden_states is None or timestep is None or encoder_hidden_states is None:
            raise ValueError(
                "hidden_states, timestep, and encoder_hidden_states are required "
                "for HunyuanImage3 denoising forward"
            )

        bsz = hidden_states.shape[0]

        # 1. Start with the full token sequence embeddings (already embedded by tokenizer)
        inputs_embeds = encoder_hidden_states

        # 2. Embed noise latents through patch_embed
        t_emb = self.time_embed(timestep)
        patch_output, token_h, token_w = self.patch_embed(hidden_states, t_emb)

        # 3. Scatter patch embeddings into image token positions
        if image_scatter_index is None and image_mask is not None:
            image_scatter_index = _indices_from_mask(image_mask)
        if image_scatter_index is not None:
            inputs_embeds = _scatter_tokens_by_index(
                inputs_embeds, image_scatter_index, patch_output
            )

        # 4. Scatter timestep embedding into timestep token positions
        if timestep_scatter_index is not None:
            n_embd = inputs_embeds.shape[-1]
            ts_emb = self.timestep_emb(timestep).reshape(bsz, 1, n_embd)
            inputs_embeds = inputs_embeds.scatter(
                dim=1,
                index=timestep_scatter_index.unsqueeze(-1).expand(-1, -1, n_embd),
                src=ts_emb,
            )

        # 5. (I2I) Scatter condition VAE image embeddings
        if cond_vae_image_mask is not None and kwargs.get("cond_vae_images") is not None:
            cond_vae_images = kwargs["cond_vae_images"].to(dtype=hidden_states.dtype)
            cond_timestep = kwargs.get("cond_timestep", timestep)
            cond_t_emb = self.time_embed(cond_timestep)
            cond_patch_output, _, _ = self.patch_embed(cond_vae_images, cond_t_emb)
            if cond_vae_scatter_index is None:
                cond_vae_scatter_index = _indices_from_mask(cond_vae_image_mask)
            inputs_embeds = _scatter_tokens_by_index(
                inputs_embeds, cond_vae_scatter_index, cond_patch_output
            )
            if kwargs.get("cond_timestep_scatter_index") is not None:
                cond_ts_index = kwargs["cond_timestep_scatter_index"]
                n_embd = inputs_embeds.shape[-1]
                cond_ts_emb = self.timestep_emb(cond_timestep).reshape(bsz, 1, n_embd)
                inputs_embeds = inputs_embeds.scatter(
                    dim=1,
                    index=cond_ts_index.unsqueeze(-1).expand(-1, -1, n_embd),
                    src=cond_ts_emb,
                )

        # 6. (I2I) Scatter condition ViT embeddings (overwrite, matching official)
        if cond_vit_image_mask is not None and cond_vit_embeds is not None:
            if cond_vit_scatter_index is None:
                cond_vit_scatter_index = _indices_from_mask(cond_vit_image_mask)
            cond_vit_embeds = cond_vit_embeds.reshape(
                bsz, -1, inputs_embeds.shape[-1]
            ).to(dtype=inputs_embeds.dtype, device=inputs_embeds.device)
            inputs_embeds = inputs_embeds.scatter(
                dim=1,
                index=cond_vit_scatter_index.to(
                    device=inputs_embeds.device, dtype=torch.long
                )
                .unsqueeze(-1)
                .expand(-1, -1, inputs_embeds.shape[-1]),
                src=cond_vit_embeds,
            )

        # 7. Apply 2D RoPE
        hidden = inputs_embeds

        # 8. Sequence Parallelism: shard hidden states across SP ranks
        forward_batch = get_forward_context().forward_batch
        sequence_shard_enabled = (
            forward_batch is not None
            and forward_batch.enable_sequence_shard
            and self.sp_size > 1
        )

        seq_len_orig = hidden.shape[1]
        seq_shard_pad = 0
        if sequence_shard_enabled:
            if seq_len_orig % self.sp_size != 0:
                seq_shard_pad = self.sp_size - (seq_len_orig % self.sp_size)
                pad = torch.zeros(
                    (bsz, seq_shard_pad, hidden.shape[2]),
                    dtype=hidden.dtype,
                    device=hidden.device,
                )
                hidden = torch.cat([hidden, pad], dim=1)

            sp_rank = get_sp_group().rank_in_group
            local_seq_len = hidden.shape[1] // self.sp_size
            hidden = hidden.view(bsz, self.sp_size, local_seq_len, hidden.shape[2])
            hidden = hidden[:, sp_rank, :, :]

        # 9. TeaCache: optionally skip the transformer layers
        self.enable_teacache = (
            forward_batch is not None and forward_batch.enable_teacache
        )

        should_skip_forward = self.should_skip_forward_for_cached_states(
            modulated_inp=t_emb,
        )

        if should_skip_forward:
            hidden = self.retrieve_cached_states(hidden)
        else:
            if self.enable_teacache:
                original_hidden = hidden.clone()

            # 9. Pass through transformer layers
            for layer in self.layers:
                hidden = layer(
                    hidden_states=hidden,
                    rope_2d=rope_2d,
                    attention_mask=encoder_attention_mask,
                )

            if self.enable_teacache:
                self.maybe_cache_states(hidden, original_hidden)

        self.cnt += 1

        # All-gather hidden states across SP ranks if sequence sharding was enabled
        if sequence_shard_enabled:
            hidden = hidden.contiguous()
            hidden = sequence_model_parallel_all_gather(hidden, dim=1)
            if seq_shard_pad > 0:
                hidden = hidden[:, :seq_len_orig, :]

        # 10. Final norm (always runs — cheap, no reason to cache)
        hidden = self.norm(hidden)

        # 11. Extract image token outputs
        if image_scatter_index is not None:
            image_output = _gather_tokens_by_index(hidden, image_scatter_index)
        else:
            image_output = hidden

        # 12. Project back to latent space via final_layer
        t_emb_2 = self.time_embed_2(timestep)
        noise_pred = self.final_layer(image_output, t_emb_2, token_h, token_w)

        return noise_pred

    # ------------------------------------------------------------------
    # TeaCache overrides (inherited from TeaCacheMixin via CachableDiT)
    # ------------------------------------------------------------------

    def should_skip_forward_for_cached_states(self, **kwargs) -> bool:
        """Check if the transformer forward can be skipped via TeaCache.

        Uses the timestep embedding (``t_emb``) as the modulated input for
        L1-distance comparison between consecutive denoising steps.
        """
        if not self.enable_teacache:
            return False
        ctx = self._get_teacache_context()
        if ctx is None:
            return False

        # Reset skip counter at the first step of each generation.
        if self.cnt == 0 and not ctx.is_cfg_negative:
            self._teacache_skipped_steps = 0

        teacache_params = ctx.teacache_params
        start_skipping, end_skipping = teacache_params.get_skip_boundaries(
            ctx.num_inference_steps, ctx.do_cfg
        )
        is_boundary_step = self.cnt < start_skipping or self.cnt >= end_skipping

        modulated_inp = kwargs["modulated_inp"]
        self.is_cfg_negative = ctx.is_cfg_negative

        should_calc = self._compute_teacache_decision(
            modulated_inp=modulated_inp,
            is_boundary_step=is_boundary_step,
            coefficients=ctx.coefficients,
            teacache_thresh=ctx.teacache_thresh,
        )
        skipped = not should_calc

        # Log skip statistics at the last step (positive branch only).
        if skipped:
            self._teacache_skipped_steps += 1
        if (
            not self.is_cfg_negative
            and self.cnt == ctx.num_inference_steps - 1
        ):
            logger.info(
                "TeaCache: skipped %d/%d denoising steps (%.1f%%)",
                self._teacache_skipped_steps,
                ctx.num_inference_steps,
                100.0 * self._teacache_skipped_steps / ctx.num_inference_steps,
            )

        return skipped

    def maybe_cache_states(
        self, hidden_states: torch.Tensor, original_hidden_states: torch.Tensor
    ) -> None:
        """Cache residual = output - input for the current CFG branch."""
        residual = hidden_states - original_hidden_states
        if not self.is_cfg_negative:
            self.previous_residual = residual
        else:
            self.previous_residual_negative = residual

    def retrieve_cached_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Reuse cached residual to skip the full transformer forward."""
        if not self.is_cfg_negative:
            return hidden_states + self.previous_residual
        else:
            return hidden_states + self.previous_residual_negative

    @torch.no_grad()
    def generate_text(
        self,
        input_ids: torch.Tensor,
        rope_2d: tuple[torch.Tensor, torch.Tensor],
        position_ids: torch.Tensor | None = None,
        max_new_tokens: int = 2048,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 1,
        eos_token_id: int | None = None,
        stop_token_ids: list[int] | None = None,
        logits_processor=None,
        progress_log_interval: int = 16,
        cond_vae_images: torch.Tensor | None = None,
        cond_timestep: torch.Tensor | None = None,
        cond_vae_scatter_index: torch.Tensor | None = None,
        cond_vit_embeds: torch.Tensor | None = None,
        cond_vit_scatter_index: torch.Tensor | None = None,
        cond_timestep_scatter_index: torch.Tensor | None = None,
        image_infos=None,
        prefill_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Autoregressive text generation using the transformer + lm_head.

        AR mode uses a different forward path than DiT mode:
        - Does NOT use patch_embed / final_layer / timestep_embed
        - Uses causal attention mask (not bidirectional)
        - Output goes through norm + lm_head to get logits

        The causal mask is built internally — step 0 gets a full
        lower-triangular bool mask, subsequent steps pass None (single
        token with KV cache needs no mask).

        Note: AR generation temporarily disables sequence parallelism because
        causal attention masks are incompatible with SP's sequence sharding.
        All TP ranks generate identical tokens (greedy by default), so no
        cross-rank synchronization is needed beyond the existing TP all-reduce.

        Args:
            input_ids: Token IDs [1, seq_len].
            rope_2d: Tuple of (cos, sin) for 2D RoPE.
            position_ids: Position IDs [1, seq_len].
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature.
            top_p: Top-p (nucleus) sampling threshold.
            top_k: Top-k sampling (1 = greedy).
            eos_token_id: End-of-sequence token ID.
            stop_token_ids: Token IDs that stop generation.
            logits_processor: Optional callable(logits, input_ids) -> logits.
            progress_log_interval: Emit progress logs every N generated tokens.
                Set to 0 to disable progress logs.

        Returns:
            Generated token IDs [1, seq_len + num_generated].
        """
        # Temporarily disable sequence parallelism for AR generation.
        # Causal attention masks require full-sequence visibility, which is
        # incompatible with SP's sequence sharding. This follows the pattern
        # used by other models (e.g., vllm-omni separates AR/DiT stages).
        original_skip_sp_values = {}
        for layer_idx, layer in enumerate(self.layers):
            attn = layer.self_attn
            if hasattr(attn, "attn") and hasattr(attn.attn, "skip_sequence_parallel"):
                original_skip_sp_values[layer_idx] = attn.attn.skip_sequence_parallel
                attn.attn.skip_sequence_parallel = True

        try:
            return self._generate_text_impl(
                input_ids=input_ids,
                rope_2d=rope_2d,
                position_ids=position_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                eos_token_id=eos_token_id,
                stop_token_ids=stop_token_ids,
                logits_processor=logits_processor,
                progress_log_interval=progress_log_interval,
                cond_vae_images=cond_vae_images,
                cond_timestep=cond_timestep,
                cond_vae_scatter_index=cond_vae_scatter_index,
                cond_vit_embeds=cond_vit_embeds,
                cond_vit_scatter_index=cond_vit_scatter_index,
                cond_timestep_scatter_index=cond_timestep_scatter_index,
                image_infos=image_infos,
                prefill_attention_mask=prefill_attention_mask,
            )
        finally:
            # Restore original skip_sequence_parallel values
            for layer_idx, original_value in original_skip_sp_values.items():
                self.layers[layer_idx].self_attn.attn.skip_sequence_parallel = original_value

    def _generate_text_impl(
        self,
        input_ids: torch.Tensor,
        rope_2d: tuple[torch.Tensor, torch.Tensor],
        position_ids: torch.Tensor | None = None,
        max_new_tokens: int = 2048,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 1,
        eos_token_id: int | None = None,
        stop_token_ids: list[int] | None = None,
        logits_processor=None,
        progress_log_interval: int = 16,
        cond_vae_images: torch.Tensor | None = None,
        cond_timestep: torch.Tensor | None = None,
        cond_vae_scatter_index: torch.Tensor | None = None,
        cond_vit_embeds: torch.Tensor | None = None,
        cond_vit_scatter_index: torch.Tensor | None = None,
        cond_timestep_scatter_index: torch.Tensor | None = None,
        image_infos=None,
        prefill_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Internal implementation of AR text generation (SP already disabled)."""
        device = input_ids.device
        start_time = time.perf_counter()
        input_seq_len = input_ids.shape[1]
        generated = input_ids.clone()
        if max_new_tokens <= 0:
            return generated

        total_seq_len = generated.shape[1] + max_new_tokens
        progress_log_interval = max(0, int(progress_log_interval or 0))
        stop_token_id_set = set(stop_token_ids or [])
        logger.info(
            "AR text generation started: prompt_tokens=%d, max_new_tokens=%d, "
            "cache_max_len=%d, stop_tokens=%d, top_k=%d, top_p=%.3f, "
            "temperature=%.3f, progress_interval=%d",
            input_seq_len,
            max_new_tokens,
            total_seq_len,
            len(stop_token_id_set),
            top_k,
            top_p,
            temperature,
            progress_log_interval,
        )
        kv_caches = HunyuanImage3ARCache(
            len(self.layers), max_cache_len=total_seq_len
        )
        initial_position_ids = None
        if position_ids is not None:
            initial_position_ids = position_ids.to(device=device, dtype=torch.long)
            if initial_position_ids.dim() == 2:
                initial_position_ids = initial_position_ids[0]
        if rope_2d is not None and total_seq_len > rope_2d[0].shape[1]:
            n_elem = rope_2d[0].shape[-1]
            rope_2d = build_batch_2d_rope(
                seq_len=total_seq_len,
                n_elem=n_elem,
                image_infos=image_infos,
                device=device,
                base=self.rope_theta,
            )

        for step in range(max_new_tokens):
            if step == 0:
                seq_len = generated.shape[1]
                current_input_ids = generated
                cache_end = seq_len
                if (
                    initial_position_ids is not None
                    and initial_position_ids.numel() >= seq_len
                ):
                    cache_position = initial_position_ids[:seq_len]
                else:
                    cache_position = torch.arange(
                        seq_len, device=device, dtype=torch.long
                    )
                # ti2i uses a hybrid mask (causal text + bidirectional image
                # region); t2i / default falls back to plain causal.
                if prefill_attention_mask is not None:
                    causal_mask = prefill_attention_mask[:, :, :seq_len, :seq_len]
                else:
                    causal_mask = torch.ones(
                        seq_len, seq_len, device=device, dtype=torch.bool
                    ).tril(diagonal=0)
                    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
                cur_rope_2d = (
                    (rope_2d[0][:, :seq_len], rope_2d[1][:, :seq_len])
                    if rope_2d is not None
                    else None
                )
            else:
                seq_pos = generated.shape[1] - 1
                current_input_ids = generated[:, -1:]
                cache_end = seq_pos + 1
                if (
                    initial_position_ids is not None
                    and initial_position_ids.numel() > seq_pos
                ):
                    cache_position = initial_position_ids[seq_pos : seq_pos + 1]
                else:
                    cache_position = torch.tensor(
                        [seq_pos], device=device, dtype=torch.long
                    )
                causal_mask = None
                cur_rope_2d = (
                    (
                        rope_2d[0][:, seq_pos : seq_pos + 1],
                        rope_2d[1][:, seq_pos : seq_pos + 1],
                    )
                    if rope_2d is not None
                    else None
                )

            outputs = self(
                input_ids=current_input_ids,
                encoder_attention_mask=causal_mask,
                rope_2d=cur_rope_2d,
                mode="ar_text",
                kv_caches=kv_caches,
                cache_position=cache_position,
                cache_end=cache_end,
                use_cache=True,
                # Inject cond vision features only at prefill (step 0): the
                # image tokens live in the initial prefix; their KV is then
                # cached, so decode steps pass no cond kwargs.
                cond_vae_images=cond_vae_images if step == 0 else None,
                cond_timestep=cond_timestep if step == 0 else None,
                cond_vae_scatter_index=cond_vae_scatter_index,
                cond_vit_embeds=cond_vit_embeds if step == 0 else None,
                cond_vit_scatter_index=cond_vit_scatter_index,
                cond_timestep_scatter_index=cond_timestep_scatter_index,
            )
            logits, kv_caches = outputs

            # 1. Apply logits processor (all ranks have identical logits after
            # All-Reduce, so applying on every rank is safe and deterministic).
            if logits_processor is not None:
                logits = logits_processor(logits, generated)

            # 2. Sample next token.  All TP ranks hold identical logits after
            # All-Reduce.  With greedy (top_k == 1 or temperature <= 0) every
            # rank independently picks the same token via argmax, so no
            # cross-rank broadcast is needed.  For non-greedy sampling, only
            # rank 0 draws a token and broadcasts it to avoid RNG divergence.
            if top_k == 1 or temperature <= 0:
                next_token = logits.argmax(dim=-1, keepdim=True)  # [1, 1]
                next_token_id = int(next_token.item())
            else:
                # Non-greedy: sample on rank 0 only, then broadcast.
                tp_rank = get_tensor_model_parallel_rank()
                if tp_rank == 0:
                    logits = logits / max(temperature, 1e-8)
                    top_k_val = min(top_k, logits.size(-1))
                    topk_vals, _ = torch.topk(logits, top_k_val)
                    logits[logits < topk_vals[:, -1:]] = float("-inf")
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(
                            logits, descending=True
                        )
                        cumulative_probs = torch.cumsum(
                            torch.softmax(sorted_logits, dim=-1), dim=-1
                        )
                        sorted_mask = cumulative_probs - torch.softmax(
                            sorted_logits, dim=-1
                        ) >= top_p
                        sorted_logits[sorted_mask] = float("-inf")
                        logits = torch.full_like(logits, float("-inf")).scatter(
                            1, sorted_indices, sorted_logits
                        )
                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    next_token_id = int(next_token.item())
                else:
                    next_token_id = 0  # placeholder, overwritten by broadcast
                # Broadcast the chosen token to all TP ranks
                token_tensor = torch.tensor([next_token_id], device=device)
                dist.broadcast(token_tensor, src=0)
                next_token_id = int(token_tensor.item())

            next_token = torch.tensor(
                [[next_token_id]], device=device, dtype=generated.dtype
            )
            generated = torch.cat([generated, next_token], dim=1)

            # 3. Check stopping conditions
            stop_reason = None
            if eos_token_id is not None and next_token_id == eos_token_id:
                stop_reason = "eos"
            elif next_token_id in stop_token_id_set:
                stop_reason = "stop_token"

            generated_tokens = step + 1
            should_log_progress = (
                progress_log_interval > 0
                and (
                    generated_tokens == 1
                    or generated_tokens % progress_log_interval == 0
                    or stop_reason is not None
                )
            )
            if should_log_progress and _is_rank0():
                elapsed = time.perf_counter() - start_time
                logger.info(
                    "AR text generation progress: generated_tokens=%d/%d, "
                    "elapsed=%.2fs, tokens_per_s=%.3f, last_token=%d, "
                    "stop_reason=%s",
                    generated_tokens,
                    max_new_tokens,
                    elapsed,
                    generated_tokens / elapsed if elapsed > 0 else 0.0,
                    next_token_id,
                    stop_reason or "none",
                )

            if stop_reason is not None:
                break

        elapsed = time.perf_counter() - start_time
        generated_tokens = generated.shape[1] - input_seq_len
        logger.info(
            "AR text generation finished: prompt_tokens=%d, generated_tokens=%d, "
            "cache_max_len=%d, elapsed=%.2fs, tokens_per_s=%.3f",
            input_seq_len,
            generated_tokens,
            total_seq_len,
            elapsed,
            generated_tokens / elapsed if elapsed > 0 else 0.0,
        )
        return generated


EntryClass = HunyuanImage3DiT
