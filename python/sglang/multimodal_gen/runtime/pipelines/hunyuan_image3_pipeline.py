# SPDX-License-Identifier: Apache-2.0
"""HunyuanImage-3.0 Pipeline for sglang-diffusion.

HunyuanImage-3.0 uses a non-standard Diffusers format: config and all
component weights live at the repo root.  The pipeline loads the
checkpoint with a single streaming pass — DiT weights stream directly
from disk into ``model.load_weights()`` without materializing the full
checkpoint, while VAE / vision weights are buffered into small dicts.
Peak CPU memory ≈ 1 shard + small auxiliary dicts.

Pattern from ComfyUI pipelines' streaming weight loading and SRT's
``buffered_multi_thread_safetensors_weights_iterator`` bounded-memory
approach (``srt/model_loader/weight_utils.py:1034``).
"""

import glob
import os
from typing import Any

import torch

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.loader.utils import (
    get_param_names_mapping,
    set_default_torch_dtype,
)
from sglang.multimodal_gen.runtime.loader.weight_utils import (
    safetensors_weights_iterator,
)
from sglang.multimodal_gen.runtime.models.vaes.autoencoder_kl_hunyuanimage3 import (
    AutoencoderKLConv3D,
)
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from sglang.multimodal_gen.runtime.pipelines_core import LoRAPipeline
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import (
    DenoisingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.hunyuan_image3 import (
    HunyuanImage3BeforeDenoisingStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


def _resolve_hunyuan_image3_checkpoint(model_path: str) -> str:
    if os.path.isdir(model_path) and os.path.isfile(
        os.path.join(model_path, "config.json")
    ):
        return model_path

    from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import (
        maybe_download_model,
    )

    return maybe_download_model(model_path)


def _list_hunyuan_image3_safetensors(model_path: str) -> list[str]:
    """List all safetensors files in *model_path* (flat checkpoint layout)."""
    safetensors_files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    if not safetensors_files:
        raise ValueError(f"No safetensors files found in {model_path}")
    return safetensors_files


# ---------------------------------------------------------------------------
# Streaming weight routing (ref: ComfyUI pipelines, SRT weight_utils)
# ---------------------------------------------------------------------------

# Prefixes that belong to VAE / vision encoder, NOT the transformer.
_NON_TRANSFORMER_PREFIXES = ("vae.", "vision_model.", "vision_aligner.")


def _route_checkpoint_weights(
    safetensors_files: list[str],
    non_transformer_prefixes: tuple[str, ...] = _NON_TRANSFORMER_PREFIXES,
) -> tuple:
    """Stream checkpoint weights, routing DiT vs auxiliary (VAE/vision).

    Returns a (dit_generator, auxiliary_dict) pair where:
    - ``dit_generator`` yields raw (name, tensor) for transformer weights
    - ``auxiliary_dict`` is {"vae": {name: tensor}, "vision_model": {...}, ...}
      and is populated as dit_generator is consumed.

    Peak CPU memory ≈ 1 shard at a time + auxiliary dicts (small components).
    The auxiliary dicts are small (~hundreds of MB) because VAE/vision are
    much smaller than the DiT (~20B params).

    Refs:
    - ComfyUI pipelines' streaming: ``comfyui_flux_pipeline.py:158``
    - SRT bounded-memory loader: ``srt/model_loader/weight_utils.py:1034``
    """
    auxiliary: dict[str, dict[str, torch.Tensor]] = {}

    def _dit_stream():
        # Stream shard-by-shard; VAE/vision weights buffered, DiT yielded.
        for name, tensor in safetensors_weights_iterator(safetensors_files):
            for prefix in non_transformer_prefixes:
                if name.startswith(prefix):
                    key = prefix.rstrip(".")
                    aux = auxiliary.get(key)
                    if aux is None:
                        aux = {}
                        auxiliary[key] = aux
                    # Strip prefix before storing.
                    aux[name[len(prefix) :]] = tensor
                    break
            else:
                # Not auxiliary → yield to DiT loader (streaming).
                yield name, tensor

    return _dit_stream(), auxiliary


# ---------------------------------------------------------------------------
# Per-component loaders
# ---------------------------------------------------------------------------


def _load_dit_model(
    model_cls,
    dit_config,
    hf_config: dict[str, Any],
    weight_stream,
    device: torch.device,
    server_args: ServerArgs,
) -> torch.nn.Module:
    """Load the DiT model with streaming weights (ref: ComfyUI pipelines).

    ``weight_stream`` is a generator yielding (name, tensor) for transformer
    weights only.  Names are in the ORIGINAL checkpoint format so that
    ``param_names_mapping`` regexes match correctly.
    """
    param_dtype = torch.bfloat16
    with set_default_torch_dtype(param_dtype):
        model = model_cls(config=dit_config, hf_config=hf_config)

    param_names_mapping_fn = get_param_names_mapping(model.param_names_mapping)
    weight_converter = getattr(model, "convert_checkpoint_weight_for_loading", None)

    def _mapped_weight_stream():
        for name, tensor in weight_stream:
            target_name, _, _ = param_names_mapping_fn(name)
            if weight_converter is not None:
                tensor = weight_converter(
                    source_name=name,
                    target_name=target_name,
                    tensor=tensor,
                )
            yield target_name, tensor

    model.load_weights(_mapped_weight_stream())
    model.post_load_weights()
    return model.eval()


def _load_vae(
    pipeline_config,
    vae_weights: dict[str, torch.Tensor],
    device: torch.device,
    local_model_path: str,
) -> torch.nn.Module:
    """Load the VAE from pre-filtered weights dict."""
    from sglang.multimodal_gen.configs.models.vaes.hunyuan_image3 import (
        HunyuanImage3VAEConfig,
    )

    vae_config = pipeline_config.vae_config
    if not isinstance(vae_config, HunyuanImage3VAEConfig):
        vae_config = HunyuanImage3VAEConfig()

    vae = AutoencoderKLConv3D(vae_config).to(device=device, dtype=torch.float32)
    if vae_weights:
        vae.load_state_dict(vae_weights, strict=False)
    logger.info("Loaded VAE from %s", local_model_path)
    return vae.eval()


def _load_image_encoder(
    hf_config: dict[str, Any],
    auxiliary_weights: dict[str, dict[str, torch.Tensor]],
    device: torch.device,
) -> dict[str, Any]:
    """Load the vision encoder and aligner from pre-filtered weights dicts."""
    vit_config = hf_config.get("vit", None)
    if vit_config is None:
        raise ValueError(
            "HunyuanImage3 image encoder was requested, but config.json does "
            "not contain a 'vit' section."
        )

    from sglang.multimodal_gen.runtime.models.encoders.siglip2 import (
        LightProjector,
        Siglip2VisionTransformer,
    )
    from transformers import Siglip2ImageProcessorFast

    vision_model = Siglip2VisionTransformer(vit_config).to(
        device=device, dtype=torch.bfloat16
    )
    vision_weights = auxiliary_weights.get("vision_model", {})
    if vision_weights:
        vision_model.load_state_dict(vision_weights, strict=False)

    vision_aligner = None
    vit_aligner_config = hf_config.get("vit_aligner", None)
    if vit_aligner_config is not None:
        vision_aligner = LightProjector(vit_aligner_config).to(
            device=device, dtype=torch.bfloat16
        )
        aligner_weights = auxiliary_weights.get("vision_aligner", {})
        if aligner_weights:
            vision_aligner.load_state_dict(aligner_weights, strict=False)

    for module in (vision_model, vision_aligner):
        if module is None:
            continue
        module.eval()
    logger.info("Loaded vision encoder components")

    return {
        "vision_model": vision_model,
        "vision_aligner": vision_aligner,
        "vit_processor": Siglip2ImageProcessorFast.from_dict(
            hf_config["vit_processor"]
        )
        if hf_config.get("vit_processor") is not None
        else None,
    }


# ---------------------------------------------------------------------------
# Pipeline class
# ---------------------------------------------------------------------------


class HunyuanImage3Pipeline(LoRAPipeline, ComposedPipelineBase):
    pipeline_name = "HunyuanImage3Pipeline"

    _required_config_modules = [
        "tokenizer",
        "transformer",
        "vae",
        "scheduler",
    ]

    def load_modules(
        self,
        server_args: ServerArgs,
        loaded_modules: dict[str, torch.nn.Module] | None = None,
    ) -> dict[str, Any]:
        """Load HunyuanImage-3.0 components with streaming weight loading.

        HY3 stores all component weights in one set of safetensors files
        (non-standard layout).  We use a single streaming pass:
        - DiT weights stream directly from disk into ``model.load_weights()``
        - VAE / vision weights are buffered into small dicts

        Peak CPU memory ≈ 1 shard + auxiliary dicts (small components).
        Follows the same streaming pattern as ``TransformerLoader`` and
        ``TextEncoderLoader`` in the sglang loader framework.
        """
        from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import (
            get_diffusers_component_config,
        )
        from transformers import AutoTokenizer

        local_model_path = _resolve_hunyuan_image3_checkpoint(server_args.model_path)
        hf_config = get_diffusers_component_config(local_model_path)
        safetensors_files = _list_hunyuan_image3_safetensors(local_model_path)
        device = get_local_torch_device()

        components: dict[str, Any] = {}

        # Tokenizer
        components["tokenizer"] = AutoTokenizer.from_pretrained(
            local_model_path,
            trust_remote_code=server_args.trust_remote_code,
        )

        # DiT weights stream from disk; VAE/vision buffered into small dicts.
        dit_stream, auxiliary = _route_checkpoint_weights(safetensors_files)

        from sglang.multimodal_gen.runtime.models.registry import ModelRegistry

        transformer_config = dict(hf_config)
        cls_name = transformer_config.pop(
            "_class_name", "HunyuanImage3DiT",
        )
        dit_config = server_args.pipeline_config.dit_config
        dit_config.update_model_arch(transformer_config)
        model_cls, _ = ModelRegistry.resolve_model_cls(cls_name)

        components["transformer"] = _load_dit_model(
            model_cls=model_cls,
            dit_config=dit_config,
            hf_config=transformer_config,
            weight_stream=dit_stream,
            device=device,
            server_args=server_args,
        )

        # VAE (from small buffered dict)
        components["vae"] = _load_vae(
            server_args.pipeline_config,
            vae_weights=auxiliary.get("vae", {}),
            device=device,
            local_model_path=local_model_path,
        )

        # Scheduler
        flow_shift = getattr(server_args.pipeline_config, "flow_shift", 3.0)
        components["scheduler"] = FlowMatchEulerDiscreteScheduler(
            shift=flow_shift,
        )

        # Image encoder (for TI2I).
        # HY3 bundles vision_model/vision_aligner in the same checkpoint,
        # so auto-detect from auxiliary weights rather than requiring an
        # explicit --image-encoder-path.  Also honour the explicit flag so
        # users can override the default.
        _has_vision_weights = bool(auxiliary.get("vision_model"))
        _explicitly_requested = "image_encoder" in server_args.component_paths
        if _has_vision_weights or _explicitly_requested:
            components["image_encoder"] = _load_image_encoder(
                hf_config,
                auxiliary_weights=auxiliary,
                device=device,
            )

        # auxiliary dict released automatically when this frame exits.
        logger.info(
            "HunyuanImage3Pipeline loaded all components from %s",
            local_model_path,
        )
        return components

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------

    def create_pipeline_stages(self, server_args: ServerArgs):
        image_encoder = self.get_module("image_encoder", None)
        self.add_stage(
            HunyuanImage3BeforeDenoisingStage(
                vae=self.get_module("vae"),
                transformer=self.get_module("transformer"),
                tokenizer=self.get_module("tokenizer"),
                scheduler=self.get_module("scheduler"),
                image_encoder=image_encoder,
            ),
            "hunyuan_image3_before_denoising_stage",
        )

        self.add_stage(
            DenoisingStage(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
            ),
        )

        self.add_standard_decoding_stage()


EntryClass = [HunyuanImage3Pipeline]
