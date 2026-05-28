# SPDX-License-Identifier: Apache-2.0
"""HunyuanImage-3.0 Pipeline for sglang-diffusion.

HunyuanImage-3.0 uses a non-standard Diffusers format: config and all
component weights live at the repo root.  The pipeline therefore loads the
flat checkpoint directly instead of using Diffusers component subdirectories.
"""

import glob
import os
from itertools import chain
from typing import Any

import torch
from torch.distributed import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.loader.fsdp_load import (
    load_model_from_full_model_state_dict,
    shard_model,
)
from sglang.multimodal_gen.runtime.loader.utils import (
    get_param_names_mapping,
    set_default_torch_dtype,
)
from sglang.multimodal_gen.runtime.loader.weight_utils import (
    default_weight_loader,
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
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.utils import set_mixed_precision_policy

logger = init_logger(__name__)


def _hunyuan_image3_transformer_weight_iterator(
    safetensors_list,
    model,
    param_names_mapping,
):
    """Yield only flat-checkpoint weights that belong to the DiT transformer."""
    valid_target_names = set(model.state_dict().keys())
    weight_converter = getattr(model, "convert_checkpoint_weight_for_loading", None)
    skipped = 0

    for name, tensor in safetensors_weights_iterator(safetensors_list):
        target_name, _, _ = param_names_mapping(name)
        if target_name in valid_target_names:
            if weight_converter is not None:
                tensor = weight_converter(
                    source_name=name,
                    target_name=target_name,
                    tensor=tensor,
                )
            yield name, tensor
        else:
            skipped += 1

    if skipped:
        logger.info(
            "Skipped %d non-transformer tensor(s) while loading HunyuanImage3 DiT",
            skipped,
        )


def _load_hunyuan_image3_transformer(
    *,
    model_cls,
    init_params: dict[str, Any],
    safetensors_list: list[str],
    device: torch.device,
    server_args: ServerArgs,
) -> torch.nn.Module:
    """Load HunyuanImage3 DiT from a flat repo checkpoint.

    The repo-root safetensors include transformer, VAE, and optional vision
    tensors. Filtering here keeps the generic FSDP loader strict while allowing
    this model-specific flat layout.
    """
    param_dtype = torch.bfloat16
    reduce_dtype = torch.float32
    mp_policy = MixedPrecisionPolicy(
        param_dtype,
        reduce_dtype,
        None,
        cast_forward_inputs=False,
    )
    set_mixed_precision_policy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        output_dtype=None,
        mp_policy=mp_policy,
    )

    with set_default_torch_dtype(param_dtype), torch.device("meta"):
        model = model_cls(**init_params)

    if server_args.use_fsdp_inference and not current_platform.is_mps():
        model._pre_fsdp_weight_loader_params = {
            n: p
            for n, p in model.named_parameters()
            if getattr(p, "weight_loader", None)
        }
        device_mesh = init_device_mesh(
            current_platform.device_type,
            mesh_shape=(server_args.hsdp_replicate_dim, server_args.hsdp_shard_dim),
            mesh_dim_names=("replicate", "shard"),
        )
        shard_model(
            model,
            cpu_offload=server_args.dit_cpu_offload,
            reshard_after_forward=True,
            mp_policy=mp_policy,
            mesh=device_mesh,
            fsdp_shard_conditions=getattr(model, "_fsdp_shard_conditions", None),
            pin_cpu_memory=server_args.pin_cpu_memory,
        )

    param_names_mapping_fn = get_param_names_mapping(model.param_names_mapping)
    weight_iterator = _hunyuan_image3_transformer_weight_iterator(
        safetensors_list,
        model,
        param_names_mapping_fn,
    )
    layerwise_cpu_materialize = (
        server_args.is_dit_layerwise_offload_selected
        and not server_args.use_fsdp_inference
    )
    if layerwise_cpu_materialize:
        logger.info(
            "Loading HunyuanImage3 DiT weights on CPU for layerwise offload "
            "initialization."
        )
    load_model_from_full_model_state_dict(
        model,
        weight_iterator,
        device,
        param_dtype,
        strict=False,
        cpu_offload=server_args.dit_cpu_offload or layerwise_cpu_materialize,
        param_names_mapping=param_names_mapping_fn,
    )

    model.post_load_weights()

    for name, param in chain(model.named_parameters(), model.named_buffers()):
        if param.is_meta:
            raise RuntimeError(f"Unexpected param or buffer {name} on meta device.")
    return model


def _resolve_hunyuan_image3_model_path(model_path: str) -> str:
    if os.path.isdir(model_path) and os.path.isfile(
        os.path.join(model_path, "config.json")
    ):
        return model_path

    from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import (
        maybe_download_model,
    )

    return maybe_download_model(model_path)


def _load_prefixed_parameters(
    module: torch.nn.Module,
    safetensors_files: list[str],
    prefix: str,
) -> int:
    if hasattr(module, "load_weights"):
        stripped_weights = (
            (name[len(prefix) :], tensor)
            for name, tensor in safetensors_weights_iterator(safetensors_files)
            if name.startswith(prefix)
        )
        loaded = module.load_weights(stripped_weights)
        return len(loaded)

    params = dict(module.named_parameters())
    loaded: set[str] = set()

    for name, tensor in safetensors_weights_iterator(safetensors_files):
        if not name.startswith(prefix):
            continue
        target_name = name[len(prefix) :]
        if target_name in params:
            default_weight_loader(params[target_name], tensor)
            loaded.add(target_name)

    missing = len(params) - len(loaded)
    if missing:
        logger.warning(
            "Loaded %d/%d parameter(s) for prefix %s; %d missing",
            len(loaded),
            len(params),
            prefix,
            missing,
        )
    return len(loaded)


def _load_hunyuan_image3_vae(
    pipeline_config,
    safetensors_files: list[str],
    device: torch.device,
    local_model_path: str,
) -> torch.nn.Module:
    from sglang.multimodal_gen.configs.models.vaes.hunyuan_image3 import (
        HunyuanImage3VAEConfig,
    )

    vae_config = pipeline_config.vae_config
    if not isinstance(vae_config, HunyuanImage3VAEConfig):
        vae_config = HunyuanImage3VAEConfig()

    vae = AutoencoderKLConv3D(vae_config).to(device=device, dtype=torch.float32)
    loaded = _load_prefixed_parameters(vae, safetensors_files, "vae.")
    logger.info("Loaded %d VAE parameter(s) from %s", loaded, local_model_path)
    return vae


def _load_hunyuan_image3_image_encoder(
    hf_config: dict[str, Any],
    safetensors_files: list[str],
    device: torch.device,
) -> dict[str, Any] | None:
    from sglang.multimodal_gen.runtime.models.encoders.siglip2 import (
        LightProjector,
        Siglip2VisionTransformer,
    )

    vit_config = hf_config.get("vit", None)
    if vit_config is None:
        return None

    vision_model = Siglip2VisionTransformer(vit_config).to(
        device=device, dtype=torch.bfloat16
    )
    vision_aligner = None
    vit_aligner_config = hf_config.get("vit_aligner", None)
    if vit_aligner_config is not None:
        vision_aligner = LightProjector(vit_aligner_config).to(
            device=device, dtype=torch.bfloat16
        )

    loaded_vision = _load_prefixed_parameters(
        vision_model, safetensors_files, "vision_model."
    )
    loaded_aligner = (
        _load_prefixed_parameters(vision_aligner, safetensors_files, "vision_aligner.")
        if vision_aligner is not None
        else 0
    )
    logger.info(
        "Loaded %d vision_model and %d vision_aligner parameter(s)",
        loaded_vision,
        loaded_aligner,
    )
    return {
        "vision_model": vision_model,
        "vision_aligner": vision_aligner,
        "vit_processor_config": hf_config.get("vit_processor", None),
    }


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
        """Load HunyuanImage-3.0 components from the flat checkpoint layout."""
        from sglang.multimodal_gen.runtime.loader.transformer_load_utils import (
            resolve_transformer_safetensors_to_load,
        )
        from sglang.multimodal_gen.runtime.models.registry import ModelRegistry
        from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import (
            get_diffusers_component_config,
        )
        from transformers import AutoTokenizer

        local_model_path = _resolve_hunyuan_image3_model_path(server_args.model_path)
        hf_config = get_diffusers_component_config(local_model_path)
        safetensors_files = sorted(
            glob.glob(os.path.join(local_model_path, "*.safetensors"))
        )
        device = get_local_torch_device()

        components: dict[str, Any] = {}

        if loaded_modules and "tokenizer" in loaded_modules:
            components["tokenizer"] = loaded_modules["tokenizer"]
        else:
            components["tokenizer"] = AutoTokenizer.from_pretrained(local_model_path)

        if loaded_modules and "transformer" in loaded_modules:
            components["transformer"] = loaded_modules["transformer"]
        else:
            transformer_config = dict(hf_config)
            transformer_config.setdefault("_class_name", "HunyuanImage3DiT")
            dit_config = server_args.pipeline_config.dit_config
            dit_config.update_model_arch(transformer_config)

            cls_name = transformer_config.pop("_class_name")
            model_cls, _ = ModelRegistry.resolve_model_cls(cls_name)
            safetensors_list = resolve_transformer_safetensors_to_load(
                server_args, local_model_path
            )
            components["transformer"] = _load_hunyuan_image3_transformer(
                model_cls=model_cls,
                init_params={"config": dit_config, "hf_config": transformer_config},
                safetensors_list=safetensors_list,
                device=device,
                server_args=server_args,
            )
            server_args.model_loaded["transformer"] = True

        if loaded_modules and "vae" in loaded_modules:
            components["vae"] = loaded_modules["vae"]
        else:
            components["vae"] = _load_hunyuan_image3_vae(
                server_args.pipeline_config,
                safetensors_files,
                device,
                local_model_path,
            )
            server_args.model_loaded["vae"] = True

        flow_shift = getattr(server_args.pipeline_config, "flow_shift", 3.0)
        components["scheduler"] = FlowMatchEulerDiscreteScheduler(
            shift=flow_shift,
        )

        task_type = getattr(server_args.pipeline_config, "task_type", None)
        if task_type is not None and task_type.accepts_image_input():
            components["image_encoder"] = _load_hunyuan_image3_image_encoder(
                hf_config,
                safetensors_files,
                device,
            )

        logger.info(
            "HunyuanImage3Pipeline loaded all components from %s",
            local_model_path,
        )
        return components

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------

    def create_pipeline_stages(self, server_args: ServerArgs):
        # AR generation is consolidated into BeforeDenoisingStage (Hybrid style).
        # When batch.bot_task is set, AR runs as the first sub-step inside
        # BeforeDenoisingStage; no separate stage is needed.
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
