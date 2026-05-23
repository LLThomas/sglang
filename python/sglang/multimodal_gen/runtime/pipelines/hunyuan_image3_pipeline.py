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
from typing import Any

import torch
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.models.vaes.autoencoder_kl_hunyuanimage3 import (
    AutoencoderKLConv3D,
)
from sglang.multimodal_gen.runtime.pipelines_core import LoRAPipeline
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages import DenoisingStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.hunyuan_image3 import (
    HunyuanImage3BeforeDenoisingStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


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
                TransformerLoader,
                resolve_transformer_safetensors_to_load,
            )
            from sglang.multimodal_gen.runtime.loader.fsdp_load import (
                maybe_load_fsdp_model,
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

            model = maybe_load_fsdp_model(
                model_cls=model_cls,
                init_params={"config": dit_config, "hf_config": config},
                weight_dir_list=safetensors_list,
                device=device,
                hsdp_replicate_dim=server_args.hsdp_replicate_dim,
                hsdp_shard_dim=server_args.hsdp_shard_dim,
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                output_dtype=None,
                cpu_offload=server_args.dit_cpu_offload,
                pin_cpu_memory=server_args.pin_cpu_memory,
                fsdp_inference=server_args.use_fsdp_inference,
                strict=False,
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
            DenoisingStage(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
            ),
        )

        self.add_standard_decoding_stage()


EntryClass = [HunyuanImage3Pipeline]