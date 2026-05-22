# SPDX-License-Identifier: Apache-2.0
"""Bridge between multimodal_gen and SRT's MoE infrastructure.

This module allows diffusion models (e.g. HunyuanImage3) to reuse SRT's
high-performance MoE stack (FusedMoE, TopK, DeepEP, ascend_fuseep, etc.)
without pulling in the full SRT server lifecycle.

- ``init_srt_parallel_groups``: creates the *minimum* SRT-side process
  groups that FusedMoE needs.  It reuses SRT's ``init_distributed_environment``
  (safe when torch.distributed is already initialised) but does **not** call
  ``initialize_model_parallel``, because that creates many extra HCCL/NCCL
  process groups (``_ATTN_CP``, ``_ATTN_TP``, ``_MOE_DP``, ``_PP``, …)
  unused by MoE that exhaust the HCCL communication-domain limit on NPUs.
- ``init_srt_moe_config``: sets the SRT global MoE configuration variables
  so that FusedMoE selects the correct kernel / dispatcher.
"""

import logging
from typing import List

import torch.distributed

logger = logging.getLogger(__name__)

_srt_parallel_groups_initialized = False
_srt_moe_config_initialized = False


# ---------------------------------------------------------------------------
# Rank-layout helpers (same logic as SRT's initialize_model_parallel)
# ---------------------------------------------------------------------------

def _tp_ranks(world_size: int, tp_size: int) -> List[List[int]]:
    return [list(range(g * tp_size, (g + 1) * tp_size))
            for g in range(world_size // tp_size)]


def _moe_ep_ranks(world_size: int, tp_size: int,
                  moe_ep_size: int, moe_dp_size: int) -> List[List[int]]:
    moe_tp_size = tp_size // moe_ep_size // moe_dp_size
    groups: List[List[int]] = []
    for g in range(world_size // tp_size):
        for dp in range(moe_dp_size):
            for t in range(moe_tp_size):
                st = g * tp_size + dp * moe_ep_size * moe_tp_size + t
                groups.append(list(range(st, st + moe_ep_size * moe_tp_size, moe_tp_size)))
    return groups


def _moe_tp_ranks(world_size: int, tp_size: int,
                  moe_ep_size: int, moe_dp_size: int) -> List[List[int]]:
    moe_tp_size = tp_size // moe_ep_size // moe_dp_size
    groups: List[List[int]] = []
    for g in range(world_size // tp_size):
        for ed in range(moe_ep_size * moe_dp_size):
            st = g * tp_size + ed * moe_tp_size
            groups.append(list(range(st, st + moe_tp_size)))
    return groups


# ---------------------------------------------------------------------------
# Lightweight group for size-1 parallel dimensions
# ---------------------------------------------------------------------------

class _SingletonGroup:
    """A lightweight stand-in for SRT's GroupCoordinator when a parallel
    dimension equals 1 (e.g. ``moe_tp_size=1``).

    ``FusedMoE.__init__`` only reads ``.world_size`` and ``.rank_in_group``
    from ``get_moe_ep_group()`` / ``get_moe_tp_group()`` — it never
    performs collective communication on these groups.  Using a
    ``_SingletonGroup`` avoids creating a useless ``torch.distributed.new_group``
    (and the associated HCCL/NCCL communication domain) for every rank.
    """

    __slots__ = ("rank", "ranks", "world_size", "rank_in_group")

    def __init__(self, rank: int):
        self.rank = rank
        self.ranks = [rank]
        self.world_size = 1
        self.rank_in_group = 0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def init_srt_parallel_groups(tp_size: int, ep_size: int) -> None:
    """Initialise the minimum SRT parallel groups for MoE.

    Must be called **after** ``torch.distributed`` has been initialised by
    multimodal_gen.

    Creates ``_WORLD``, ``_TP``, ``_MOE_EP``, ``_MOE_TP`` — the four
    GroupCoordinators that FusedMoE actually references.  The rank layout
    matches SRT's ``initialize_model_parallel`` with
    ``pipeline_model_parallel_size=1, moe_data_model_parallel_size=1``.

    For size-1 dimensions (e.g. ``moe_tp_size=1``), a lightweight
    ``_SingletonGroup`` is used instead of a full ``GroupCoordinator``,
    avoiding the creation of unused HCCL/NCCL communication domains.
    """
    global _srt_parallel_groups_initialized
    if _srt_parallel_groups_initialized:
        return

    import sglang.srt.distributed.parallel_state as srt_ps
    from sglang.srt.distributed.parallel_state import (
        init_distributed_environment,
        init_model_parallel_group,
    )
    from sglang.multimodal_gen.runtime.platforms import current_platform

    backend = current_platform.get_torch_distributed_backend_str()
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()

    # _WORLD — reuse SRT's init_distributed_environment; it skips
    # init_process_group when torch.distributed is already initialised.
    init_distributed_environment(backend=backend)

    # _TP
    srt_ps._TP = init_model_parallel_group(
        _tp_ranks(world_size, tp_size),
        srt_ps._WORLD.local_rank, backend,
        use_pynccl=False, use_custom_allreduce=False, group_name="tp",
    )

    # _ATTN_TP — srt's VisionAttention (e.g. SigLIP2 in HunyuanImage3 ti2i)
    # calls get_attention_tp_size()/get_attn_tp_group() at __init__, which
    # require _ATTN_TP to be set.  srt aliases _ATTN_TP = _TP when
    # attn_tp_size == tp_size (parallel_state.py:2000); multimodal_gen uses
    # the same size, so mirror that here.
    srt_ps._ATTN_TP = srt_ps._TP

    # MoE dimensions
    moe_ep_size = ep_size
    moe_dp_size = 1  # multimodal_gen does not use MoE DP
    moe_tp_size = tp_size // moe_ep_size // moe_dp_size

    # _MOE_EP — alias _TP when ep_size == tp_size, _SingletonGroup when
    # ep_size == 1, otherwise create a real GroupCoordinator.
    if moe_ep_size == tp_size:
        srt_ps._MOE_EP = srt_ps._TP
    elif moe_ep_size == 1:
        srt_ps._MOE_EP = _SingletonGroup(rank)
    else:
        srt_ps._MOE_EP = init_model_parallel_group(
            _moe_ep_ranks(world_size, tp_size, moe_ep_size, moe_dp_size),
            srt_ps._WORLD.local_rank, backend,
            use_pynccl=False, use_custom_allreduce=False, group_name="moe_ep",
        )

    # _MOE_TP — alias _TP when moe_tp_size == tp_size, _SingletonGroup
    # when moe_tp_size == 1, otherwise create a real GroupCoordinator.
    if moe_tp_size == tp_size:
        srt_ps._MOE_TP = srt_ps._TP
    elif moe_tp_size == 1:
        srt_ps._MOE_TP = _SingletonGroup(rank)
    else:
        srt_ps._MOE_TP = init_model_parallel_group(
            _moe_tp_ranks(world_size, tp_size, moe_ep_size, moe_dp_size),
            srt_ps._WORLD.local_rank, backend,
            use_pynccl=False, use_custom_allreduce=False, group_name="moe_tp",
        )

    _srt_parallel_groups_initialized = True
    logger.info(
        "SRT MoE parallel groups initialised: tp=%d, ep=%d (moe_ep=%d, moe_tp=%d), backend=%s",
        tp_size, ep_size, moe_ep_size, moe_tp_size, backend,
    )


def init_srt_moe_config() -> None:
    """Set SRT's global MoE configuration variables.

    Reads ``moe_a2a_backend`` and ``moe_runner_backend`` from the
    multimodal_gen global server args and writes them into SRT's
    ``MOE_A2A_BACKEND`` / ``MOE_RUNNER_BACKEND`` globals.

    Also constructs a minimal SRT ``ServerArgs`` and sets it as the
    SRT global server args, because SRT's MoE code (FusedMoE, TopK,
    dispatchers, etc.) widely calls ``get_global_server_args()``.

    Called lazily when the first MoE layer is constructed.
    """
    global _srt_moe_config_initialized
    if _srt_moe_config_initialized:
        return

    from sglang.multimodal_gen.runtime.server_args import get_global_server_args

    server_args = get_global_server_args()

    # Set SRT's MoE globals directly.
    import sglang.srt.layers.moe.utils as moe_utils
    from sglang.srt.layers.moe.utils import DeepEPMode, MoeA2ABackend, MoeRunnerBackend

    moe_utils.MOE_A2A_BACKEND = MoeA2ABackend(server_args.moe_a2a_backend)
    moe_utils.MOE_RUNNER_BACKEND = MoeRunnerBackend(server_args.moe_runner_backend)
    moe_utils.DEEPEP_MODE = DeepEPMode.AUTO
    moe_utils.DEEPEP_CONFIG = ""

    # Minimal SRT ServerArgs for get_global_server_args() calls.
    from sglang.srt.server_args import (
        ServerArgs as SrtServerArgs,
        set_global_server_args_for_scheduler,
    )

    srt_server_args = SrtServerArgs(model_path="dummy")
    srt_server_args.moe_a2a_backend = server_args.moe_a2a_backend
    srt_server_args.moe_runner_backend = server_args.moe_runner_backend
    set_global_server_args_for_scheduler(srt_server_args)

    # Diffusion models always run in extend (prefill) mode.
    from sglang.srt.layers.dp_attention import _DpGatheredBufferWrapper
    _DpGatheredBufferWrapper.set_is_extend_in_batch(True)

    _srt_moe_config_initialized = True
    logger.info(
        "SRT MoE config initialised: a2a_backend=%s, runner_backend=%s",
        server_args.moe_a2a_backend, server_args.moe_runner_backend,
    )
