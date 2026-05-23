from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.sample.sampling_params import DataType, SamplingParams


@dataclass
class HunyuanImage3SamplingParams(SamplingParams):
    num_inference_steps: int = 50
    height: int = 1024
    width: int = 1024
    guidance_scale: float = 5.0
    seed: int = 42
    cot_text: str = ""  # DiT-only 模式：由调用者提供 CoT 文本
    data_type: DataType = DataType.IMAGE

    # AR stage params
    bot_task: str = ""  # "", "think", "recaption", "think_recaption"
    sequence_template: str = "pretrain"  # "pretrain" or "instruct"
    system_prompt: str = ""  # system prompt (auto-selected by bot_task if empty)
    ar_max_new_tokens: int = 2048  # max tokens for AR generation
    ar_temperature: float = 1.0  # AR sampling temperature
    ar_top_p: float = 1.0  # AR top-p
    ar_top_k: int = 1  # AR top-k (1 = greedy)
    drop_think: bool = False  # drop think portion from CoT
    image_size: str = "auto"  # "auto" = predict ratio via AR, or "HxW"
    sys_type: str = "dynamic"  # system prompt type: "None", "en_vanilla", "en_recaption", "en_think_recaption", "en_unified", "dynamic", "custom"