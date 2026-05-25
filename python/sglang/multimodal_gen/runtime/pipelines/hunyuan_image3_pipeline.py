# SPDX-License-Identifier: Apache-2.0
"""HunyuanImage-3.0 Pipeline for sglang-diffusion.

HunyuanImage-3.0 uses a non-standard Diffusers format (transformers auto_map
with config.json instead of model_index.json).  All weights live in the repo
root as sharded safetensors; there are no sub-directories for transformer/vae.

This pipeline therefore overrides ``_load_config()`` and ``load_modules()`` to
handle the custom layout, similar to ``Hunyuan3D2Pipeline``.
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


def _apply_hunyuan_image3_npu_overrides(server_args: ServerArgs) -> None:
    if (
        current_platform.is_npu()
        and server_args.use_fsdp_inference
        and server_args.dit_cpu_offload
        and server_args.pin_cpu_memory
    ):
        logger.warning(
            "Disabling pin_cpu_memory for HunyuanImage3 DiT FSDP CPU offload on NPU. "
            "Pinned H2D parameter copies can trigger Ascend vector core exceptions "
            "during repeated denoising unshards."
        )
        server_args.pin_cpu_memory = False


def _hunyuan_image3_transformer_weight_iterator(
    safetensors_list,
    model,
    param_names_mapping,
):
    """Yield only flat-checkpoint weights that belong to the DiT transformer."""
    valid_target_names = set(model.state_dict().keys())
    skipped = 0

    for name, tensor in safetensors_weights_iterator(safetensors_list):
        target_name, _, _ = param_names_mapping(name)
        if target_name in valid_target_names:
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
    load_model_from_full_model_state_dict(
        model,
        weight_iterator,
        device,
        param_dtype,
        strict=False,
        cpu_offload=server_args.dit_cpu_offload,
        param_names_mapping=param_names_mapping_fn,
    )

    model.post_load_weights()

    for name, param in chain(model.named_parameters(), model.named_buffers()):
        if param.is_meta:
            raise RuntimeError(f"Unexpected param or buffer {name} on meta device.")
    return model


class HunyuanImage3Pipeline(LoRAPipeline, ComposedPipelineBase):
    pipeline_name = "HunyuanImage3ForCausalMM"

    _required_config_modules = [
        "tokenizer",
        "transformer",
        "vae",
        "scheduler",
    ]

    # ------------------------------------------------------------------
    # Custom model layout: no model_index.json
    # ------------------------------------------------------------------

    def _load_config(self) -> dict[str, Any]:
        """Return a synthetic model_index for the non-Diffusers repo layout.

        HunyuanImage-3.0 ships a ``config.json`` (transformers format) instead
        of ``model_index.json``.  We synthesize the index so that
        ``ComposedPipelineBase.load_modules()`` can dispatch to the correct
        component loaders.
        """
        return {
            "_class_name": self.pipeline_name,
            "_diffusers_version": "0.0.0",
            "tokenizer": ["transformers", "AutoTokenizer"],
            "transformer": ["diffusers", "HunyuanImage3DiT"],
            "vae": ["diffusers", "AutoencoderKLConv3D"],
            "scheduler": [
                "diffusers",
                "FlowMatchEulerDiscreteScheduler",
            ],
        }

    def load_modules(
        self,
        server_args: ServerArgs,
        loaded_modules: dict[str, torch.nn.Module] | None = None,
    ) -> dict[str, Any]:
        """Load all components for the HunyuanImage-3.0 model.

        Because the model uses a non-standard layout (no sub-directories), we
        handle each component manually instead of relying on the default
        ``ComposedPipelineBase.load_modules()`` which expects Diffusers
        sub-directories.
        """
        from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import (
            maybe_download_model,
        )

        model_path = server_args.model_path
        device = get_local_torch_device()
        _apply_hunyuan_image3_npu_overrides(server_args)

        # Download the full model if not already local
        if not os.path.isdir(model_path) or not os.path.isfile(
            os.path.join(model_path, "config.json")
        ):
            local_model_path = maybe_download_model(model_path)
        else:
            local_model_path = model_path

        components: dict[str, Any] = {}

        # --- Tokenizer ---
        if loaded_modules and "tokenizer" in loaded_modules:
            components["tokenizer"] = loaded_modules["tokenizer"]
        else:
            from transformers import AutoTokenizer

            components["tokenizer"] = AutoTokenizer.from_pretrained(
                local_model_path,
            )

        # --- Transformer (DiT) ---
        if loaded_modules and "transformer" in loaded_modules:
            components["transformer"] = loaded_modules["transformer"]
        else:
            from sglang.multimodal_gen.runtime.loader.component_loaders.transformer_loader import (
                resolve_transformer_safetensors_to_load,
            )
            from sglang.multimodal_gen.runtime.models.registry import ModelRegistry
            from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import (
                get_diffusers_component_config,
            )

            # Read HF config and inject _class_name for ModelRegistry
            config = get_diffusers_component_config(local_model_path)
            if "_class_name" not in config:
                config["_class_name"] = "HunyuanImage3DiT"

            # Update dit_config from HF config before instantiation
            dit_config = server_args.pipeline_config.dit_config
            dit_config.update_model_arch(config)

            # Resolve model class
            cls_name = config.pop("_class_name")
            model_cls, _ = ModelRegistry.resolve_model_cls(cls_name)

            # Find safetensors files
            safetensors_list = resolve_transformer_safetensors_to_load(
                server_args, local_model_path
            )

            model = _load_hunyuan_image3_transformer(
                model_cls=model_cls,
                init_params={"config": dit_config, "hf_config": config},
                safetensors_list=safetensors_list,
                device=device,
                server_args=server_args,
            )
            components["transformer"] = model
            server_args.model_loaded["transformer"] = True

        # --- VAE ---
        if loaded_modules and "vae" in loaded_modules:
            components["vae"] = loaded_modules["vae"]
        else:
            from sglang.multimodal_gen.configs.models.vaes.hunyuan_image3 import (
                HunyuanImage3VAEConfig,
            )
            from sglang.multimodal_gen.runtime.loader.weight_utils import (
                default_weight_loader,
                safetensors_weights_iterator,
            )

            vae_config = server_args.pipeline_config.vae_config
            if not isinstance(vae_config, HunyuanImage3VAEConfig):
                vae_config = HunyuanImage3VAEConfig()

            vae = AutoencoderKLConv3D(vae_config)

            # Load VAE weights from the shared safetensors files.
            # HunyuanImage-3.0 stores all weights (transformer + vae + ...)
            # in the same sharded safetensors at the repo root, with "vae."
            # prefix for VAE parameters.
            safetensors_files = sorted(
                glob.glob(os.path.join(local_model_path, "*.safetensors"))
            )
            if safetensors_files:
                vae = vae.to(device=device, dtype=torch.float32)
                params_dict = dict(vae.named_parameters())
                loaded_params: set[str] = set()

                for name, tensor in safetensors_weights_iterator(safetensors_files):
                    # Only load VAE weights (prefixed with "vae.")
                    if not name.startswith("vae."):
                        continue
                    # Strip "vae." prefix
                    name = name[4:]
                    if name in params_dict:
                        default_weight_loader(params_dict[name], tensor)
                        loaded_params.add(name)

                logger.info(
                    "Loaded %d/%d VAE parameters from %s",
                    len(loaded_params),
                    len(params_dict),
                    local_model_path,
                )

            components["vae"] = vae
            server_args.model_loaded["vae"] = True

        # --- Scheduler ---
        flow_shift = getattr(
            server_args.pipeline_config, "flow_shift", 3.0
        )
        components["scheduler"] = FlowMatchEulerDiscreteScheduler(
            shift=flow_shift,
        )

        # --- Image Encoder (SigLIP2 + LightProjector) for TI2I ---
        # Only load if the pipeline config requires image input (TI2I mode)
        task_type = getattr(server_args.pipeline_config, "task_type", None)
        if task_type is not None and task_type.accepts_image_input():
            from sglang.multimodal_gen.runtime.loader.weight_utils import (
                default_weight_loader,
                safetensors_weights_iterator,
            )
            from sglang.multimodal_gen.runtime.models.encoders.siglip2 import (
                LightProjector,
                Siglip2VisionTransformer,
            )

            # Read HF config for vit and vit_aligner configs
            from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import (
                get_diffusers_component_config,
            )

            hf_config = get_diffusers_component_config(local_model_path)

            # Load SigLIP2 vision model
            vit_config = hf_config.get("vit", None)
            vit_aligner_config = hf_config.get("vit_aligner", None)

            image_encoder = None
            if vit_config is not None:
                vision_model = Siglip2VisionTransformer(vit_config)
                vision_aligner = None
                if vit_aligner_config is not None:
                    vision_aligner = LightProjector(vit_aligner_config)

                # Load weights from shared safetensors
                safetensors_files = sorted(
                    glob.glob(os.path.join(local_model_path, "*.safetensors"))
                )
                if safetensors_files:
                    vision_model = vision_model.to(device=device, dtype=torch.bfloat16)
                    if vision_aligner is not None:
                        vision_aligner = vision_aligner.to(
                            device=device, dtype=torch.bfloat16
                        )

                    vm_params = dict(vision_model.named_parameters())
                    va_params = (
                        dict(vision_aligner.named_parameters())
                        if vision_aligner is not None
                        else {}
                    )
                    loaded_vm: set[str] = set()
                    loaded_va: set[str] = set()

                    for name, tensor in safetensors_weights_iterator(safetensors_files):
                        if name.startswith("vision_model."):
                            stripped = name[len("vision_model."):]
                            if stripped in vm_params:
                                default_weight_loader(vm_params[stripped], tensor)
                                loaded_vm.add(stripped)
                        elif name.startswith("vision_aligner."):
                            stripped = name[len("vision_aligner."):]
                            if stripped in va_params:
                                default_weight_loader(va_params[stripped], tensor)
                                loaded_va.add(stripped)

                    logger.info(
                        "Loaded %d/%d vision_model, %d/%d vision_aligner parameters",
                        len(loaded_vm),
                        len(vm_params),
                        len(loaded_va),
                        len(va_params),
                    )

                image_encoder = {
                    "vision_model": vision_model,
                    "vision_aligner": vision_aligner,
                    "vit_processor_config": hf_config.get("vit_processor", None),
                }

            components["image_encoder"] = image_encoder

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
