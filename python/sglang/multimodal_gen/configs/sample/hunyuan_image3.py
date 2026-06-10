import json
import os
from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.sample.sampling_params import DataType, SamplingParams
from sglang.multimodal_gen.configs.sample.teacache import TeaCacheParams


@dataclass
class HunyuanImage3SamplingParams(SamplingParams):
    num_inference_steps: int = 50
    height: int = 1024
    width: int = 1024
    guidance_scale: float = 5.0
    seed: int = 42
    data_type: DataType = DataType.IMAGE
    image_size: str = ""  # "HxW" format, e.g. "1024x1024"

    # AR stage params — will be added in Phase 2
    # bot_task, ar_max_new_tokens, ar_temperature, ar_top_p, ar_top_k,
    # drop_think, system_prompt, sys_type, sequence_template, cot_text

    # TeaCache params for denoising acceleration.
    # Coefficients calibrated via polyfit on 3920 data points (80 prompts × 49 steps)
    # from vllm-omni's HunyuanImage3 TeaCache config.
    # Set ``enable_teacache=True`` via the API or JSON config to activate.
    teacache_params: TeaCacheParams = field(
        default_factory=lambda: TeaCacheParams(
            teacache_thresh=0.2,
            coefficients=[
                1.04117826e02,
                -1.26848482e02,
                5.68168652e01,
                -1.04182570e01,
                6.78098549e-01,
            ],
        )
    )

    def _adjust(self, server_args):
        self._apply_generation_config_defaults(server_args)
        super()._adjust(server_args)

    def _apply_generation_config_defaults(self, server_args):
        """Apply official HunyuanImage3 generation defaults when not overridden."""
        explicit_fields = getattr(self, "_explicit_fields", set()) or set()
        if not getattr(server_args, "model_path", None):
            return
        config_path = os.path.join(server_args.model_path, "generation_config.json")
        if not os.path.exists(config_path):
            return

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                generation_config = json.load(f)
        except Exception:
            return

        field_map = {
            "diff_infer_steps": "num_inference_steps",
            "diff_guidance_scale": "guidance_scale",
        }
        for config_key, field_name in field_map.items():
            if (
                field_name not in explicit_fields
                and config_key in generation_config
                and generation_config[config_key] is not None
            ):
                setattr(self, field_name, generation_config[config_key])
