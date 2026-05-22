"""HunyuanImage-3.0 BeforeDenoisingStage and Tokenizer.

This module implements the preprocessing logic for HunyuanImage-3.0 DiT-only mode,
including token sequence construction and all preprocessing steps needed before
the standard DenoisingStage.
"""

import random
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from diffusers.utils.torch_utils import randn_tensor

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.models.dits.hunyuan_image3 import (
    HUNYUAN_IMAGE3_EXTRA_RESOLUTIONS,
    Resolution,
    ResolutionGroup,
    build_batch_2d_rope,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Tokenizer output dataclass
# ---------------------------------------------------------------------------


@dataclass
class TokenizerOutput:
    """Output from HunyuanImage3Tokenizer sequence building."""

    tokens: torch.Tensor  # [seq_len] token IDs
    image_mask: torch.Tensor  # [seq_len] bool mask for <img> positions
    timestep_scatter_index: torch.Tensor | None  # positions of all <timestep> tokens
    gen_timestep_scatter_index: torch.Tensor | None  # positions of gen <timestep> tokens
    cond_timestep_scatter_index: torch.Tensor | None  # positions of cond <timestep> tokens
    guidance_scatter_index: torch.Tensor | None  # positions of <guidance> tokens
    cond_vae_image_mask: torch.Tensor | None  # mask for conditional VAE image (I2I)
    cond_vit_image_mask: torch.Tensor | None  # mask for conditional ViT image (I2I)
    text_mask: torch.Tensor | None  # float mask for text positions
    gen_image_slices: list | None  # slice objects for gen image regions
    all_image_slices: list | None  # slice objects for all image regions


# ---------------------------------------------------------------------------
# HunyuanImage3Tokenizer
# ---------------------------------------------------------------------------


class HunyuanImage3Tokenizer:
    """Tokenizer wrapper for HunyuanImage-3.0 DiT-only mode.

    Builds the unified token sequence (text + image placeholders + special tokens)
    that the single-stream transformer processes.
    """

    def __init__(self, tokenizer):
        if isinstance(tokenizer, str):
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.tokenizer = tokenizer

        # Extract special token IDs
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.boi_token_id = tokenizer.convert_tokens_to_ids("<boi>")
        self.eoi_token_id = tokenizer.convert_tokens_to_ids("<eoi>")
        self.img_token_id = tokenizer.convert_tokens_to_ids("<img>")
        self.cfg_token_id = tokenizer.convert_tokens_to_ids("<cfg>")
        self.timestep_token_id = tokenizer.convert_tokens_to_ids("<timestep>")
        self.guidance_token_id = tokenizer.convert_tokens_to_ids("<guidance>")
        self.joint_img_sep_token_id = tokenizer.convert_tokens_to_ids(
            "<joint_img_sep>"
        )
        self.special_token_map = tokenizer.added_tokens_encoder

        # Instruct template tokens
        self.answer_token_id = self.special_token_map.get("<answer>", None)
        self.end_of_answer_token_id = self.special_token_map.get("</answer>", None)
        self.think_token_id = self.special_token_map.get("ჼ", None)  # think start
        self.end_of_think_token_id = self.special_token_map.get(
            "ebil_think", None
        )  # think end
        self.recaption_token_id = self.special_token_map.get("<recaption>", None)
        self.end_of_recaption_token_id = self.special_token_map.get(
            "</recaption>", None
        )

    def encode_text(self, text, uncond_p=0.0, max_length=None):
        """Encode text, optionally replacing with <cfg> tokens for unconditional."""
        do_uncond = (uncond_p == 1.0) or (uncond_p > 0 and random.random() < uncond_p)

        if isinstance(text, str):
            text_tokens = self.tokenizer.encode(text, add_special_tokens=False)
        else:
            text_tokens = list(text)

        if do_uncond:
            text_tokens = [self.cfg_token_id] * len(text_tokens)

        if max_length is not None and len(text_tokens) > max_length:
            text_tokens = text_tokens[:max_length]

        return text_tokens

    def _add_image_meta_tokens(
        self,
        token_seq,
        token_count,
        extra_token_pos,
        add_timestep_token=True,
        add_guidance_token=False,
        add_image_shape_token=True,
        base_size=None,
        ratio_idx=None,
        image_type="gen_image",
    ):
        """Add image metadata tokens (size, ratio, timestep, guidance)."""
        if add_image_shape_token and base_size is not None and ratio_idx is not None:
            token_seq.extend(
                [
                    self.special_token_map[f"<img_size_{base_size}>"],
                    self.special_token_map[f"<img_ratio_{ratio_idx}>"],
                ]
            )
            token_count += 2
        if add_timestep_token:
            token_seq.append(self.special_token_map["<timestep>"])
            extra_token_pos["timestep"].append(token_count)
            if image_type == "gen_image":
                extra_token_pos["gen_timestep"].append(token_count)
            elif image_type == "joint_image":
                extra_token_pos["cond_timestep"].append(token_count)
            token_count += 1
        if add_guidance_token:
            token_seq.append(self.special_token_map["<guidance>"])
            extra_token_pos["guidance"].append(token_count)
            token_count += 1
        return token_count

    def build_t2i_sequence(
        self,
        prompt,
        token_h,
        token_w,
        base_size=1024,
        ratio_idx=0,
        cot_text="",
        system_prompt="",
        uncond_p=0.0,
        template="pretrain",
    ):
        """Build the T2I token sequence.

        Args:
            template: "pretrain" or "instruct".
                pretrain: <bos> [system] [user] [cot] <boi> <img_size> <img_ratio>
                          <timestep> <img>*N <eoi> <eos>
                instruct: <bos> [system]\n\n User: [user]\n\n Assistant: [cot]
                          <answer> <boi> <img_size> <img_ratio> <timestep>
                          <img>*N <eoi> </answer> <eos>
        """
        uncond_kwargs = dict(uncond_p=uncond_p)

        sections = []

        if template == "instruct":
            # Instruct template: add role prefixes and answer tags
            if system_prompt:
                sections.append(dict(type="text", text=system_prompt))
            sections.append(
                dict(type="text", text=f"\n\n User: {prompt}\n\n Assistant: ", **uncond_kwargs)
            )
            if cot_text:
                sections.append(dict(type="text", text=cot_text, **uncond_kwargs))
            # <answer> before image
            if self.answer_token_id is not None:
                sections.append(
                    dict(type="text", text="<answer>", ignore=True, **uncond_kwargs)
                )
        else:
            # Pretrain template: no role prefixes
            if system_prompt:
                sections.append(dict(type="text", text=system_prompt))
            sections.append(dict(type="text", text=prompt, **uncond_kwargs))
            if cot_text:
                sections.append(dict(type="text", text=cot_text, **uncond_kwargs))

        # Generated image
        image_token_length = token_h * token_w
        sections.append(
            dict(
                type="gen_image",
                token_length=image_token_length,
                add_timestep_token=True,
                add_guidance_token=False,
                use_front_boi_token=True,
                add_image_shape_token=True,
                base_size=base_size,
                ratio_idx=ratio_idx,
            )
        )

        # </answer> after image (instruct template)
        if template == "instruct" and self.end_of_answer_token_id is not None:
            sections.append(
                dict(type="text", text="</answer>", ignore=True, **uncond_kwargs)
            )

        return self._encode_sections(sections)

    def build_uncond_sequence(
        self,
        prompt,
        token_h,
        token_w,
        base_size=1024,
        ratio_idx=0,
        cot_text="",
        system_prompt="",
        template="pretrain",
    ):
        """Build the unconditional token sequence for CFG.

        Uses two-pass encoding: re-encodes with uncond_p=1.0, which replaces
        all text tokens with <cfg> tokens. This matches the official approach
        where the unconditional sequence is built by a fresh encoding pass.
        """
        return self.build_t2i_sequence(
            prompt=prompt,
            token_h=token_h,
            token_w=token_w,
            base_size=base_size,
            ratio_idx=ratio_idx,
            cot_text=cot_text,
            system_prompt=system_prompt,
            uncond_p=1.0,
            template=template,
        )

    def _encode_sections(self, sections):
        """Encode a list of sections into a token sequence."""
        template = "-".join([s["type"] for s in sections])
        sections = deepcopy(sections)

        token_source = defaultdict(list)
        text_mask_specs = []

        for section in sections:
            if section["type"] == "text":
                text = self.encode_text(
                    section["text"],
                    uncond_p=section.get("uncond_p", 0.0),
                )
                token_source["text"].append(text)
                text_mask_specs.append(
                    dict(
                        ignore=section.get("ignore", False),
                        start_offset=section.get("start_offset", 0),
                        end_offset=section.get("end_offset", 0),
                    )
                )
            elif section["type"] == "gen_image":
                token_source["gen_image"].append(
                    dict(
                        length=section["token_length"],
                        timestep=section.get("add_timestep_token", True),
                        guidance=section.get("add_guidance_token", False),
                        front_boi=section.get("use_front_boi_token", True),
                        image_shape=section.get("add_image_shape_token", True),
                        base_size=section.get("base_size"),
                        ratio_idx=section.get("ratio_idx"),
                    )
                )
            elif section["type"] == "joint_image":
                token_source["joint_image"].append(
                    dict(
                        length=section["token_length"],
                        timestep=section.get("add_timestep_token", True),
                        front_boi=section.get("use_front_boi_token", True),
                        image_shape=section.get("add_image_shape_token", True),
                        base_size=section.get("base_size"),
                        ratio_idx=section.get("ratio_idx"),
                    )
                )

        # Build the full token sequence
        full_token_seq, extra_token_pos = self._build_token_sequence(
            template=template,
            token_source=dict(token_source),
        )

        full_seq_tensor = torch.tensor(full_token_seq, dtype=torch.long)

        # Parse masks from extra_token_pos
        timestep_scatter_index = (
            torch.tensor(extra_token_pos["timestep"], dtype=torch.long)
            if "timestep" in extra_token_pos
            else None
        )
        gen_timestep_scatter_index = (
            torch.tensor(extra_token_pos["gen_timestep"], dtype=torch.long)
            if "gen_timestep" in extra_token_pos
            else None
        )
        cond_timestep_scatter_index = (
            torch.tensor(extra_token_pos["cond_timestep"], dtype=torch.long)
            if "cond_timestep" in extra_token_pos
            else None
        )
        guidance_scatter_index = (
            torch.tensor(extra_token_pos["guidance"], dtype=torch.long)
            if "guidance" in extra_token_pos
            else None
        )

        # Gen image mask
        gen_image_slices, gen_image_mask = self._parse_image_slices(
            extra_token_pos, "img", full_seq_tensor
        )

        # All image slices
        all_image_slices = (
            [
                slice(s, e + 1)
                for s, e in zip(
                    extra_token_pos["<all_img>_start"],
                    extra_token_pos["<all_img>_end"],
                )
            ]
            if "<all_img>_start" in extra_token_pos
            and "<all_img>_end" in extra_token_pos
            else []
        )

        # Conditional vae/vit image masks (for I2I)
        _, cond_vae_image_mask = self._parse_image_slices(
            extra_token_pos, "vae_img", full_seq_tensor
        )
        _, cond_vit_image_mask = self._parse_image_slices(
            extra_token_pos, "vit_img", full_seq_tensor
        )

        # Text mask
        text_slices = (
            [
                slice(s, e + 1)
                for s, e in zip(
                    extra_token_pos["<text>_start"],
                    extra_token_pos["<text>_end"],
                )
            ]
            if "<text>_start" in extra_token_pos
            and "<text>_end" in extra_token_pos
            else []
        )
        if text_slices:
            text_mask = torch.zeros_like(full_seq_tensor, dtype=torch.float32)
            for text_slice, mask_spec in zip(text_slices, text_mask_specs):
                if not mask_spec["ignore"]:
                    real_slice = slice(
                        text_slice.start + mask_spec["start_offset"],
                        text_slice.stop + mask_spec["end_offset"],
                    )
                    text_mask[real_slice] = 1.0
        else:
            text_mask = None

        return TokenizerOutput(
            tokens=full_seq_tensor,
            image_mask=gen_image_mask,
            timestep_scatter_index=timestep_scatter_index,
            gen_timestep_scatter_index=gen_timestep_scatter_index,
            cond_timestep_scatter_index=cond_timestep_scatter_index,
            guidance_scatter_index=guidance_scatter_index,
            cond_vae_image_mask=cond_vae_image_mask,
            cond_vit_image_mask=cond_vit_image_mask,
            text_mask=text_mask,
            gen_image_slices=gen_image_slices,
            all_image_slices=all_image_slices,
        )

    def _build_token_sequence(
        self, template, token_source, add_bos=True, add_eos=True
    ):
        """Build the full token sequence from template and token source.

        Simplified version of vllm-omni's encode_sequence.
        """
        keys = template.split("-")
        index_indicator = {k: 0 for k in token_source}

        token_seq = []
        token_count = 0
        extra_token_pos = defaultdict(list)

        if add_bos:
            token_seq.append(self.bos_token_id)
            token_count += 1

        for key in keys:
            source = token_source[key][index_indicator[key]]

            if key == "text":
                token_seq.extend(source)
                extra_token_pos["<text>_start"].append(token_count)
                if (
                    "<cfg>_start" not in extra_token_pos
                    and len(source) > 0
                    and source[0] == self.cfg_token_id
                ):
                    extra_token_pos["<cfg>_start"].append(token_count)
                token_count += len(source)
                extra_token_pos["<text>_end"].append(token_count - 1)

            elif key == "gen_image":
                if isinstance(source, int):
                    source = {"length": source}
                # <boi>
                if source.get("front_boi", True):
                    token_seq.append(self.boi_token_id)
                    extra_token_pos["boi"].append(token_count)
                    token_count += 1
                # Image meta tokens
                token_count = self._add_image_meta_tokens(
                    token_seq=token_seq,
                    token_count=token_count,
                    extra_token_pos=extra_token_pos,
                    add_timestep_token=source.get("timestep", True),
                    add_guidance_token=source.get("guidance", False),
                    add_image_shape_token=source.get("image_shape", True),
                    base_size=source.get("base_size"),
                    ratio_idx=source.get("ratio_idx"),
                    image_type=key,
                )
                # <img>*N <eoi>
                token_seq.extend(
                    [self.img_token_id] * source["length"] + [self.eoi_token_id]
                )
                extra_token_pos["<img>_start"].append(token_count)
                extra_token_pos["<all_img>_start"].append(token_count)
                token_count += source["length"]
                extra_token_pos["<img>_end"].append(token_count - 1)
                extra_token_pos["<all_img>_end"].append(token_count - 1)
                extra_token_pos["eoi"].append(token_count)
                token_count += 1  # <eoi>

            elif key == "joint_image":
                assert isinstance(source["length"], list) and len(source["length"]) == 2
                # <boi>
                if source.get("front_boi", True):
                    token_seq.append(self.boi_token_id)
                    extra_token_pos["boi"].append(token_count)
                    token_count += 1
                # Meta tokens
                token_count = self._add_image_meta_tokens(
                    token_seq=token_seq,
                    token_count=token_count,
                    extra_token_pos=extra_token_pos,
                    add_timestep_token=source.get("timestep", True),
                    add_image_shape_token=source.get("image_shape", True),
                    base_size=source.get("base_size"),
                    ratio_idx=source.get("ratio_idx"),
                    image_type=key,
                )
                # VAE image tokens
                token_seq.extend([self.img_token_id] * source["length"][0])
                extra_token_pos["<vae_img>_start"].append(token_count)
                extra_token_pos["<joint_img>_start"].append(token_count)
                extra_token_pos["<all_img>_start"].append(token_count)
                token_count += source["length"][0]
                extra_token_pos["<vae_img>_end"].append(token_count - 1)
                extra_token_pos["<all_img>_end"].append(token_count - 1)
                # <joint_img_sep>
                token_seq.append(self.joint_img_sep_token_id)
                extra_token_pos["joint_img_sep"].append(token_count)
                token_count += 1
                # ViT image tokens
                token_seq.extend([self.img_token_id] * source["length"][1])
                extra_token_pos["<vit_img>_start"].append(token_count)
                extra_token_pos["<all_img>_start"].append(token_count)
                token_count += source["length"][1]
                extra_token_pos["<vit_img>_end"].append(token_count - 1)
                extra_token_pos["<joint_img>_end"].append(token_count - 1)
                extra_token_pos["<all_img>_end"].append(token_count - 1)
                # <eoi>
                token_seq.append(self.eoi_token_id)
                extra_token_pos["eoi"].append(token_count)
                token_count += 1

            index_indicator[key] += 1

        if add_eos:
            token_seq.append(self.eos_token_id)
            extra_token_pos["eos"].append(token_count)
            token_count += 1

        return token_seq, extra_token_pos

    @staticmethod
    def _parse_image_slices(extra_token_pos, prefix, tokens):
        """Parse image slices and mask from extra_token_pos."""
        start_key = f"<{prefix}>_start" if prefix == "img" else f"<{prefix}_start>"
        end_key = f"<{prefix}>_end" if prefix == "img" else f"<{prefix}_end>"

        image_slices = (
            [
                slice(s, e + 1)
                for s, e in zip(extra_token_pos[start_key], extra_token_pos[end_key])
            ]
            if start_key in extra_token_pos and end_key in extra_token_pos
            else []
        )
        if image_slices:
            image_mask = torch.zeros_like(tokens, dtype=torch.bool)
            for image_slice in image_slices:
                image_mask[image_slice] = True
        else:
            image_mask = None
        return image_slices, image_mask

    def build_ar_sequence(
        self,
        prompt,
        system_prompt="",
        bot_task="think",
        template="pretrain",
        base_size=1024,
    ):
        """Build the AR input token sequence (text prefix only, no image placeholders).

        pretrain: <bos> [system] [prompt] Ⴟ
        instruct: <bos> [system]\\n\\n User: [prompt]\\n\\n Assistant: Ⴟ

        Returns:
            dict with input_ids [1, seq_len], attention_mask, position_ids.
        """
        token_seq = [self.bos_token_id]

        if template == "instruct":
            if system_prompt:
                token_seq.extend(self.encode_text(system_prompt))
                token_seq.extend(self.encode_text("\n\n User: "))
            else:
                token_seq.extend(self.encode_text("User: "))
            token_seq.extend(self.encode_text(prompt))
            token_seq.extend(self.encode_text("\n\n Assistant: "))
        else:
            if system_prompt:
                token_seq.extend(self.encode_text(system_prompt))
            token_seq.extend(self.encode_text(prompt))

        # Add think start token (ჼ) if bot_task includes think
        if bot_task in ("think", "think_recaption") and self.think_token_id is not None:
            token_seq.append(self.think_token_id)

        input_ids = torch.tensor([token_seq], dtype=torch.long)
        seq_len = input_ids.shape[1]
        position_ids = torch.arange(seq_len).unsqueeze(0)

        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
        }

    def parse_cot_text(self, cot_text):
        """Parse CoT text into sections list for _encode_sections.

        Input: "Ⴟthink contentebil_think<recaption>recaption content</recaption>"
        Output: [
            dict(type="text", text="Ⴟthink contentebil_think"),
            dict(type="text", text="<recaption>recaption content</recaption>"),
        ]
        """
        if not cot_text:
            return []

        sections = []
        # Split on <recaption> boundary if present
        if "<recaption>" in cot_text:
            parts = cot_text.split("<recaption>", 1)
            think_part = parts[0]
            recaption_part = "<recaption>" + parts[1]
            if think_part:
                sections.append(dict(type="text", text=think_part))
            sections.append(dict(type="text", text=recaption_part))
        else:
            sections.append(dict(type="text", text=cot_text))

        return sections

    def get_ratio_token_ids(self):
        """Return all <img_ratio_*> token IDs as a list."""
        ratio_ids = []
        for idx in range(len(self.special_token_map)):
            key = f"<img_ratio_{idx}>"
            if key in self.special_token_map:
                ratio_ids.append(self.special_token_map[key])
            else:
                break
        return ratio_ids

    def get_ratio_index_from_token_id(self, token_id):
        """Map a ratio token ID back to ratio_index."""
        for idx in range(len(self.special_token_map)):
            key = f"<img_ratio_{idx}>"
            if key in self.special_token_map:
                if self.special_token_map[key] == token_id:
                    return idx
            else:
                break
        return 0


# ---------------------------------------------------------------------------
# Logits Processors for AR generation
# ---------------------------------------------------------------------------


class StageTransitionLogitsProcessor:
    """Force specific token sequences at stage boundaries during AR generation.

    Reference: official modeling_hunyuan_image_3.py:_StageTransitionLogitsProcessor

    When the last generated token matches a transition key, the processor
    forces the next N tokens to be the specified sequence.

    Args:
        transitions: dict mapping trigger_token_id -> list of token_ids to force.
    """

    def __init__(self, transitions):
        self.transitions = transitions
        self._forced_queue = []

    def __call__(self, logits, input_ids):
        # If we have forced tokens in the queue, apply them
        if self._forced_queue:
            forced_token_id = self._forced_queue.pop(0)
            logits[:] = float("-inf")
            logits[0, forced_token_id] = 0.0
            return logits

        # Check if the last token triggers a transition
        last_token = input_ids[0, -1].item()
        if last_token in self.transitions:
            forced_tokens = self.transitions[last_token]
            if len(forced_tokens) > 0:
                # Force the first token now, queue the rest
                first_forced = forced_tokens[0]
                self._forced_queue.extend(forced_tokens[1:])
                logits[:] = float("-inf")
                logits[0, first_forced] = 0.0

        return logits


class ConditionalSliceVocabLogitsProcessor:
    """Restrict vocabulary to a subset after a trigger token during AR generation.

    Reference: official modeling_hunyuan_image_3.py:_ConditionalSliceVocabLogitsProcessor

    When the last token matches the trigger, restrict logits to only the
    allowed token IDs (used for ratio prediction after <img_size_*>).

    Args:
        trigger_token_id: Token ID that activates the restriction.
        allowed_token_ids: List of token IDs allowed after the trigger.
    """

    def __init__(self, trigger_token_id, allowed_token_ids):
        self.trigger_token_id = trigger_token_id
        self.allowed_token_ids = allowed_token_ids

    def __call__(self, logits, input_ids):
        last_token = input_ids[0, -1].item()
        if last_token == self.trigger_token_id:
            mask = torch.full_like(logits, float("-inf"))
            mask[0, self.allowed_token_ids] = 0.0
            logits = logits + mask
        return logits


# ---------------------------------------------------------------------------
# System Prompts
# ---------------------------------------------------------------------------


SYSTEM_PROMPTS = {
    "think_recaption": (
        "You are an AI assistant capable of generating images. "
        "When given a user prompt, you should first think about the visual details, "
        "then recaption the prompt with richer visual descriptions. "
        "Output your thinking process starting with Ⴟ, end with ebil_think, "
        "then provide the recaptioned prompt within <recaption></recaption> tags."
    ),
    "recaption": (
        "You are an AI assistant capable of generating images. "
        "When given a user prompt, recaption it with richer visual descriptions. "
        "Output the recaptioned prompt within <recaption></recaption> tags."
    ),
    "think": (
        "You are an AI assistant capable of generating images. "
        "When given a user prompt, think about the visual details. "
        "Output your thinking process starting with Ⴟ, end with ebil_think."
    ),
    "vanilla": "",
}


def _resolve_system_prompt(batch, bot_task=""):
    """Auto-select system prompt based on bot_task, unless user specified one."""
    system_prompt = getattr(batch, "system_prompt", "") or ""
    if system_prompt:
        return system_prompt
    if bot_task and bot_task in SYSTEM_PROMPTS:
        return SYSTEM_PROMPTS[bot_task]
    return ""


# ---------------------------------------------------------------------------
# HunyuanImage3BeforeDenoisingStage
# ---------------------------------------------------------------------------


class HunyuanImage3BeforeDenoisingStage(PipelineStage):
    """BeforeDenoisingStage for HunyuanImage-3.0 DiT-only mode.

    Prepares all inputs needed by DenoisingStage:
    - Token sequences (conditional + unconditional) via HunyuanImage3Tokenizer
    - Token embeddings via transformer.embed_tokens
    - 2D RoPE (cos, sin) via build_batch_2d_rope
    - Initial noise latents
    - Timesteps via FlowMatchEulerDiscreteScheduler
    - All masks and scatter indices
    """

    def __init__(self, vae, transformer, tokenizer, scheduler):
        super().__init__()
        self.vae = vae
        self.transformer = transformer
        self.tokenizer_wrapper = HunyuanImage3Tokenizer(tokenizer)
        self.scheduler = scheduler
        self.resolution_group = ResolutionGroup(
            base_size=1024,
            step=64,
            align=1,
            extra_resolutions=[
                Resolution(s) for s in HUNYUAN_IMAGE3_EXTRA_RESOLUTIONS
            ],
        )

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        """Process batch and populate all fields needed by DenoisingStage."""
        device = get_local_torch_device()
        dtype = torch.bfloat16

        # 1. Parse inputs from batch
        prompt = batch.prompt
        if isinstance(prompt, list):
            prompt = prompt[0]
        cot_text = getattr(batch, "cot_text", "") or ""
        system_prompt = getattr(batch, "system_prompt", "") or ""
        template = getattr(batch, "sequence_template", "pretrain") or "pretrain"
        height = batch.height
        width = batch.width
        guidance_scale = batch.guidance_scale
        num_inference_steps = batch.num_inference_steps
        seed = batch.seed

        # 1b. If AR stage predicted ratio_index, override height/width
        if hasattr(batch, "ratio_index") and batch.ratio_index is not None:
            reso = self.resolution_group[batch.ratio_index]
            height = reso.h
            width = reso.w

        # 2. Get target resolution from bucket
        target_w, target_h = self.resolution_group.get_target_size(width, height)

        # 3. Calculate token dimensions
        dit_config = server_args.pipeline_config.dit_config.arch_config
        vae_downsample = dit_config.vae_downsample_factor[0]  # (16, 16)
        token_h = target_h // vae_downsample
        token_w = target_w // vae_downsample

        # 4. Get base_size and ratio_idx
        base_size, ratio_idx = self.resolution_group.get_base_size_and_ratio_index(
            target_w, target_h
        )

        # 5. Build conditional token sequence
        cond_output = self.tokenizer_wrapper.build_t2i_sequence(
            prompt=prompt,
            token_h=token_h,
            token_w=token_w,
            base_size=base_size,
            ratio_idx=ratio_idx,
            cot_text=cot_text,
            system_prompt=system_prompt,
            uncond_p=0.0,
            template=template,
        )

        # 6. Build unconditional token sequence (CFG) via two-pass encoding
        do_cfg = guidance_scale > 1.0
        if do_cfg:
            uncond_output = self.tokenizer_wrapper.build_uncond_sequence(
                prompt=prompt,
                token_h=token_h,
                token_w=token_w,
                base_size=base_size,
                ratio_idx=ratio_idx,
                cot_text=cot_text,
                system_prompt=system_prompt,
                template=template,
            )

        # 7. Get token embeddings via transformer.embed_tokens
        cond_tokens = cond_output.tokens.unsqueeze(0).to(device)  # [1, seq_len]
        cond_embeds = self.transformer.embed_tokens(cond_tokens)  # [1, seq_len, hidden_size]

        if do_cfg:
            uncond_tokens = uncond_output.tokens.unsqueeze(0).to(device)
            uncond_embeds = self.transformer.embed_tokens(uncond_tokens)

        # 8. Build 2D RoPE
        # Find the image region in the token sequence for RoPE
        image_infos = None
        if cond_output.all_image_slices:
            image_infos = [
                [(s, (token_h, token_w)) for s in cond_output.all_image_slices]
            ]

        seq_len = cond_tokens.shape[1]
        n_elem = dit_config.rope_axes_dim[0] + dit_config.rope_axes_dim[1]  # 128
        rope_cos, rope_sin = build_batch_2d_rope(
            seq_len=seq_len,
            n_elem=n_elem,
            image_infos=image_infos,
            device=device,
            base=dit_config.rope_theta,
        )
        rope_2d = (rope_cos, rope_sin)

        # 9. Build position IDs
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)  # [1, seq_len]

        # 10. Build hybrid attention mask (causal for text, bidirectional for image)
        # Official approach: start with lower-triangular causal mask, then set
        # image token positions to True (bidirectional within image regions).
        attention_mask = torch.ones(
            seq_len, seq_len, device=device, dtype=torch.bool
        ).tril(diagonal=0)
        # Set image regions to bidirectional attention
        if cond_output.all_image_slices:
            for img_slice in cond_output.all_image_slices:
                attention_mask[img_slice, img_slice] = True
        # Expand to [1, 1, seq_len, seq_len] for SDPA
        encoder_attention_mask = attention_mask.unsqueeze(0).unsqueeze(0).to(dtype)
        if do_cfg:
            neg_encoder_attention_mask = encoder_attention_mask.clone()

        # 11. Prepare image masks and scatter indices
        image_mask = cond_output.image_mask.unsqueeze(0).to(device)  # [1, seq_len]
        timestep_scatter_index = (
            cond_output.timestep_scatter_index.unsqueeze(0).to(device)
            if cond_output.timestep_scatter_index is not None
            else None
        )
        gen_timestep_scatter_index = (
            cond_output.gen_timestep_scatter_index.unsqueeze(0).to(device)
            if cond_output.gen_timestep_scatter_index is not None
            else None
        )
        cond_timestep_scatter_index = (
            cond_output.cond_timestep_scatter_index.unsqueeze(0).to(device)
            if cond_output.cond_timestep_scatter_index is not None
            else None
        )
        cond_vae_image_mask = (
            cond_output.cond_vae_image_mask.unsqueeze(0).to(device)
            if cond_output.cond_vae_image_mask is not None
            else None
        )
        cond_vit_image_mask = (
            cond_output.cond_vit_image_mask.unsqueeze(0).to(device)
            if cond_output.cond_vit_image_mask is not None
            else None
        )

        # 12. Prepare initial noise latents
        latent_channels = dit_config.in_channels  # 32
        generator = torch.Generator(device=device).manual_seed(seed)
        latents = randn_tensor(
            (1, latent_channels, token_h, token_w),
            generator=generator,
            device=device,
            dtype=torch.float32,
        )

        # 13. Set up scheduler timesteps
        scheduler = self.scheduler
        scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = scheduler.timesteps

        # 14. Populate batch fields
        batch.prompt_embeds = [cond_embeds]
        if do_cfg:
            batch.negative_prompt_embeds = [uncond_embeds]
        else:
            batch.negative_prompt_embeds = []

        batch.latents = latents
        batch.timesteps = timesteps
        batch.scheduler = scheduler
        batch.num_inference_steps = num_inference_steps
        batch.sigmas = scheduler.sigmas.tolist()  # Python list, not numpy
        batch.generator = generator
        batch.do_classifier_free_guidance = do_cfg

        # Masks and indices for PipelineConfig methods
        batch.encoder_attention_mask = encoder_attention_mask
        if do_cfg:
            batch.neg_encoder_attention_mask = neg_encoder_attention_mask
        batch.image_mask = image_mask
        batch.timestep_scatter_index = timestep_scatter_index
        batch.gen_timestep_scatter_index = gen_timestep_scatter_index
        batch.cond_timestep_scatter_index = cond_timestep_scatter_index
        batch.cond_vae_image_mask = cond_vae_image_mask
        batch.cond_vit_image_mask = cond_vit_image_mask
        batch.cond_vit_embeds = None  # No ViT condition for T2I
        batch.rope_2d = rope_2d
        batch.position_ids = position_ids
        batch.token_h = token_h
        batch.token_w = token_w

        batch.height = target_h
        batch.width = target_w
        batch.raw_latent_shape = latents.shape

        return batch


# ---------------------------------------------------------------------------
# HunyuanImage3TextGenerationStage
# ---------------------------------------------------------------------------


class HunyuanImage3TextGenerationStage(PipelineStage):
    """AR autoregressive text generation stage for HunyuanImage-3.0.

    Responsible for:
    1. Running AR to generate CoT text (think/recaption)
    2. Auto-predicting image ratio (ratio_index)
    3. Writing results to batch for downstream BeforeDenoisingStage

    Data flow:
      batch.bot_task -> AR generation -> batch.cot_text + batch.ratio_index
    """

    def __init__(self, transformer, tokenizer):
        super().__init__()
        self.transformer = transformer
        self.tokenizer_wrapper = HunyuanImage3Tokenizer(tokenizer)

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        """Run AR text generation if bot_task is set."""
        bot_task = getattr(batch, "bot_task", "") or ""
        if not bot_task:
            return batch  # Skip AR stage

        device = get_local_torch_device()
        dit_config = server_args.pipeline_config.dit_config.arch_config

        # 1. Read parameters
        template = getattr(batch, "sequence_template", "pretrain") or "pretrain"
        system_prompt = _resolve_system_prompt(batch, bot_task)
        base_size = dit_config.image_base_size

        # 2. Build AR input sequence
        ar_input = self.tokenizer_wrapper.build_ar_sequence(
            prompt=batch.prompt if isinstance(batch.prompt, str) else batch.prompt[0],
            system_prompt=system_prompt,
            bot_task=bot_task,
            template=template,
            base_size=base_size,
        )

        # 3. Build causal attention mask
        input_ids = ar_input["input_ids"].to(device)
        seq_len = input_ids.shape[1]
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device),
            diagonal=1,
        ).unsqueeze(0).unsqueeze(0)

        # 4. Build 2D RoPE for text (no image regions, uses 1D positions)
        n_elem = dit_config.rope_axes_dim[0] + dit_config.rope_axes_dim[1]
        rope_cos, rope_sin = build_batch_2d_rope(
            seq_len=seq_len,
            n_elem=n_elem,
            image_infos=None,
            device=device,
            base=dit_config.rope_theta,
        )
        rope_2d = (rope_cos, rope_sin)

        # 5. Configure stage transitions and logits processor
        transitions = self._build_stage_transitions(bot_task, template, base_size)
        ratio_token_ids = self.tokenizer_wrapper.get_ratio_token_ids()
        img_size_token_id = self.tokenizer_wrapper.special_token_map.get(
            f"<img_size_{base_size}>", None
        )

        processors = []
        if transitions:
            processors.append(StageTransitionLogitsProcessor(transitions))
        if img_size_token_id is not None and ratio_token_ids:
            processors.append(
                ConditionalSliceVocabLogitsProcessor(
                    trigger_token_id=img_size_token_id,
                    allowed_token_ids=ratio_token_ids,
                )
            )

        logits_processor = processors[0] if len(processors) == 1 else (
            _ComposeLogitsProcessors(processors) if len(processors) > 1 else None
        )

        # 6. Run AR generation
        generated_ids = self.transformer.generate_text(
            input_ids=input_ids,
            attention_mask=causal_mask,
            rope_2d=rope_2d,
            position_ids=ar_input["position_ids"].to(device),
            max_new_tokens=getattr(batch, "ar_max_new_tokens", 2048),
            temperature=getattr(batch, "ar_temperature", 1.0),
            top_p=getattr(batch, "ar_top_p", 1.0),
            top_k=getattr(batch, "ar_top_k", 1),
            eos_token_id=self.tokenizer_wrapper.eos_token_id,
            stop_token_ids=ratio_token_ids if ratio_token_ids else None,
            logits_processor=logits_processor,
        )

        # 7. Parse output: extract cot_text and ratio_index
        cot_text, ratio_index = self._parse_ar_output(generated_ids[0])
        batch.cot_text = cot_text
        if ratio_index is not None:
            batch.ratio_index = ratio_index

        logger.info(
            "AR stage: bot_task=%s, cot_text_len=%d, ratio_index=%s",
            bot_task,
            len(cot_text),
            ratio_index,
        )

        return batch

    def _build_stage_transitions(self, bot_task, template, base_size):
        """Build stage transitions config based on bot_task and template."""
        transitions = {}
        tw = self.tokenizer_wrapper

        if bot_task in ("think", "think_recaption"):
            # After think ends (ebil_think), force <recaption> or <boi>
            if tw.end_of_think_token_id is not None:
                if bot_task == "think_recaption" and tw.recaption_token_id is not None:
                    transitions[tw.end_of_think_token_id] = [tw.recaption_token_id]
                elif tw.boi_token_id is not None:
                    # think only: after think, go to <boi>
                    img_size_id = tw.special_token_map.get(f"<img_size_{base_size}>")
                    forced = [tw.boi_token_id]
                    if img_size_id is not None:
                        forced.append(img_size_id)
                    transitions[tw.end_of_think_token_id] = forced

        if bot_task in ("recaption", "think_recaption"):
            # After </recaption>, force <answer><boi><img_size> or <boi><img_size>
            if tw.end_of_recaption_token_id is not None:
                img_size_id = tw.special_token_map.get(f"<img_size_{base_size}>")
                if template == "instruct" and tw.answer_token_id is not None:
                    forced = [tw.answer_token_id, tw.boi_token_id]
                else:
                    forced = [tw.boi_token_id]
                if img_size_id is not None:
                    forced.append(img_size_id)
                transitions[tw.end_of_recaption_token_id] = forced

        return transitions if transitions else None

    def _parse_ar_output(self, generated_ids):
        """Parse AR output to extract cot_text and ratio_index.

        Args:
            generated_ids: 1D tensor of token IDs.

        Returns:
            (cot_text, ratio_index) tuple.
        """
        tw = self.tokenizer_wrapper
        ratio_token_ids = tw.get_ratio_token_ids()

        # Last token may be a ratio token
        ratio_index = None
        last_token = generated_ids[-1].item()
        if last_token in ratio_token_ids:
            ratio_index = tw.get_ratio_index_from_token_id(last_token)

        # Decode the generated portion (excluding input prefix) to get cot_text
        # Find where the AR input ends and generated text begins
        # The input ends with Ⴟ (think_token_id), so text after that is CoT
        input_end_idx = 0
        if tw.think_token_id is not None:
            think_positions = (generated_ids == tw.think_token_id).nonzero(as_tuple=True)[0]
            if len(think_positions) > 0:
                input_end_idx = think_positions[-1].item() + 1

        # Decode generated tokens to text
        gen_tokens = generated_ids[input_end_idx:]
        # Remove trailing ratio token from decoded text
        if ratio_index is not None and len(gen_tokens) > 0:
            gen_tokens = gen_tokens[:-1]

        cot_text = tw.tokenizer.decode(gen_tokens.tolist(), skip_special_tokens=False)

        return cot_text, ratio_index


class _ComposeLogitsProcessors:
    """Compose multiple logits processors into one."""

    def __init__(self, processors):
        self.processors = processors

    def __call__(self, logits, input_ids):
        for processor in self.processors:
            logits = processor(logits, input_ids)
        return logits