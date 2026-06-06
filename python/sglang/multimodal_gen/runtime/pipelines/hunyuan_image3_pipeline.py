# SPDX-License-Identifier: Apache-2.0
"""HunyuanImage-3.0 Pipeline for sglang-diffusion.

HunyuanImage-3.0 uses a non-standard Diffusers format: config and all
component weights live at the repo root.  The pipeline therefore loads the
flat checkpoint directly instead of using Diffusers component subdirectories.

Weight loading follows the same pattern as ``Hunyuan3DPipeline``: load all
shards into a single dict, then dispatch to each component.
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
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.hunyuan_image3 import (
    HunyuanImage3BeforeDenoisingStage,
    HunyuanImage3DenoisingStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Checkpoint resolution (ref: Hunyuan3DPipeline._resolve_shape_dir)
# ---------------------------------------------------------------------------


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
# Single-pass loading (ref: Hunyuan3DPipeline._load_and_split_checkpoint)
# ---------------------------------------------------------------------------

# Prefixes that belong to VAE / vision encoder, NOT the transformer.
_NON_TRANSFORMER_PREFIXES = ("vae.", "vision_model.", "vision_aligner.")


def _load_flat_checkpoint(
    safetensors_files: list[str],
) -> dict[str, torch.Tensor]:
    """Load all shards into a single flat dict (key → tensor).

    Preserves the ORIGINAL checkpoint key names so that the transformer's
    ``param_names_mapping`` regexes work correctly.

    Uses ``safetensors_weights_iterator`` so the standard tqdm progress bar
    is shown automatically (same as ComfyUI pipelines and base class loader).
    """
    return dict(safetensors_weights_iterator(safetensors_files))


# ---------------------------------------------------------------------------
# Per-component loaders
# ---------------------------------------------------------------------------


def _load_dit_model(
    model_cls,
    dit_config,
    hf_config: dict[str, Any],
    flat_weights: dict[str, torch.Tensor],
    device: torch.device,
    server_args: ServerArgs,
) -> torch.nn.Module:
    """Load the DiT model (ref: ``Hunyuan3DPipeline._load_dit_model``).

    HunyuanImage3's flat checkpoint stores transformer weights under
    **multiple** top-level prefixes (``model.``, ``patch_embed.``, ``lm_head.``,
    ``time_embed.``, etc.).  All of these must be passed to the transformer's
    ``load_weights`` — only ``vae.`` and ``vision_*`` keys are excluded.

    The original key names are preserved so that ``param_names_mapping``
    regexes (e.g. ``^model\\.(.*)$``) match correctly.
    """
    param_dtype = torch.bfloat16
    with set_default_torch_dtype(param_dtype):
        model = model_cls(config=dit_config, hf_config=hf_config)

    param_names_mapping_fn = get_param_names_mapping(model.param_names_mapping)
    weight_converter = getattr(model, "convert_checkpoint_weight_for_loading", None)

    def _weight_iterator():
        for name, tensor in flat_weights.items():
            # Skip VAE / vision encoder keys — they are loaded separately.
            if name.startswith(_NON_TRANSFORMER_PREFIXES):
                continue
            target_name, _, _ = param_names_mapping_fn(name)
            if weight_converter is not None:
                tensor = weight_converter(
                    source_name=name,
                    target_name=target_name,
                    tensor=tensor,
                )
            yield target_name, tensor

    model.load_weights(_weight_iterator())
    model.post_load_weights()
    return model.eval()


def _load_vae(
    pipeline_config,
    flat_weights: dict[str, torch.Tensor],
    device: torch.device,
    local_model_path: str,
) -> torch.nn.Module:
    """Load the VAE (ref: ``Hunyuan3DPipeline._load_simple_component``)."""
    from sglang.multimodal_gen.configs.models.vaes.hunyuan_image3 import (
        HunyuanImage3VAEConfig,
    )

    vae_config = pipeline_config.vae_config
    if not isinstance(vae_config, HunyuanImage3VAEConfig):
        vae_config = HunyuanImage3VAEConfig()

    vae = AutoencoderKLConv3D(vae_config).to(device=device, dtype=torch.float32)

    prefix = "vae."
    vae_weights = {
        key[len(prefix):]: tensor
        for key, tensor in flat_weights.items()
        if key.startswith(prefix)
    }
    if vae_weights:
        vae.load_state_dict(vae_weights, strict=False)
    logger.info("Loaded VAE from %s", local_model_path)
    return vae.eval()


def _load_image_encoder(
    hf_config: dict[str, Any],
    flat_weights: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, Any]:
    """Load the vision encoder and aligner (ref: ``Hunyuan3DPipeline._load_simple_component``)."""
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
    prefix = "vision_model."
    vision_weights = {
        key[len(prefix):]: tensor
        for key, tensor in flat_weights.items()
        if key.startswith(prefix)
    }
    if vision_weights:
        vision_model.load_state_dict(vision_weights, strict=False)

    vision_aligner = None
    vit_aligner_config = hf_config.get("vit_aligner", None)
    if vit_aligner_config is not None:
        vision_aligner = LightProjector(vit_aligner_config).to(
            device=device, dtype=torch.bfloat16
        )
        prefix = "vision_aligner."
        aligner_weights = {
            key[len(prefix):]: tensor
            for key, tensor in flat_weights.items()
            if key.startswith(prefix)
        }
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
    pipeline_name = "HunyuanImage3ForCausalMM"

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
        """Load HunyuanImage-3.0 components from the flat checkpoint layout.

        Follows ``Hunyuan3DPipeline.load_modules`` pattern: load all shards
        once, then dispatch to component loaders.
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

        # --- Single-pass load (ref: Hunyuan3DPipeline) ---
        flat_weights = _load_flat_checkpoint(safetensors_files)

        # Transformer
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
            flat_weights=flat_weights,
            device=device,
            server_args=server_args,
        )

        # VAE
        components["vae"] = _load_vae(
            server_args.pipeline_config,
            flat_weights=flat_weights,
            device=device,
            local_model_path=local_model_path,
        )

        # Scheduler
        flow_shift = getattr(server_args.pipeline_config, "flow_shift", 3.0)
        components["scheduler"] = FlowMatchEulerDiscreteScheduler(
            shift=flow_shift,
        )

        # Image encoder (optional, for TI2I).
        # Enabled when user passes --component-paths.image_encoder.
        # The weights are loaded from the same flat checkpoint; no separate
        # path is needed because HY3 stores all components in one shard set.
        if "image_encoder" in server_args.component_paths:
            components["image_encoder"] = _load_image_encoder(
                hf_config,
                flat_weights=flat_weights,
                device=device,
            )

        # Release checkpoint dict to free CPU memory.
        flat_weights.clear()

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
            HunyuanImage3DenoisingStage(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
            ),
        )

        self.add_standard_decoding_stage()


EntryClass = [HunyuanImage3Pipeline]
