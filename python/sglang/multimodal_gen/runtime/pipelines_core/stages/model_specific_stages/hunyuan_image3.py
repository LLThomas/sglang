"""HunyuanImage-3.0 before-denoising stage and tokenizer.

This module implements the preprocessing logic for HunyuanImage-3.0, including
token sequence construction and all preprocessing steps needed before denoising.

Architecture (Phase 1 - DiT-only T2I):
This version supports DiT-only text-to-image generation without AR support.
The AR generation features (CoT / ratio prediction) are disabled in this phase.
"""

import random
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass

import torch
import torch.distributed as dist
from diffusers.utils.torch_utils import randn_tensor

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.models.dits.hunyuan_image3 import (
    HUNYUAN_IMAGE3_EXTRA_RESOLUTIONS,
    Resolution,
    ResolutionGroup,
    build_batch_2d_rope,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentUse,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    StageValidators as V,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    VerificationResult,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


def _is_rank0() -> bool:
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def _first_prompt(prompt):
    return prompt[0] if isinstance(prompt, list) else prompt


def _compose_logits_processors(processors):
    if not processors:
        return None
    if len(processors) == 1:
        return processors[0]

    def composed(logits, input_ids):
        for processor in processors:
            logits = processor(logits, input_ids)
        return logits

    return composed


# ---------------------------------------------------------------------------
# Tokenizer output dataclass
# ---------------------------------------------------------------------------


@dataclass
class TokenizerOutput:
    """Output from HunyuanImage3Tokenizer sequence building."""

    tokens: torch.Tensor  # [seq_len] token IDs
    image_mask: torch.Tensor  # [seq_len] bool mask for <img> positions
    image_scatter_index: torch.Tensor | None  # positions of gen image tokens
    timestep_scatter_index: torch.Tensor | None  # positions of all <timestep> tokens
    gen_timestep_scatter_index: torch.Tensor | None  # positions of gen <timestep> tokens
    cond_timestep_scatter_index: torch.Tensor | None  # positions of cond <timestep> tokens
    guidance_scatter_index: torch.Tensor | None  # positions of <guidance> tokens
    cond_vae_image_mask: torch.Tensor | None  # mask for conditional VAE image (I2I)
    cond_vae_scatter_index: torch.Tensor | None  # positions of conditional VAE tokens
    cond_vit_image_mask: torch.Tensor | None  # mask for conditional ViT image (I2I)
    cond_vit_scatter_index: torch.Tensor | None  # positions of conditional ViT tokens
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

            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer, trust_remote_code=True
            )
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
        self.think_token = "larla"
        self.end_of_think_token = "elarla"
        self.recaption_token = "<recaption>"
        self.end_of_recaption_token = "</recaption>"
        self.think_token_id = self.special_token_map.get(
            self.think_token, self.special_token_map.get("n~", None)
        )
        self.end_of_think_token_id = self.special_token_map.get(
            self.end_of_think_token,
            self.special_token_map.get("ebil_think", None),
        )
        self.recaption_token_id = self.special_token_map.get("<recaption>", None)
        self.end_of_recaption_token_id = self.special_token_map.get(
            "</recaption>", None
        )

        conv = getattr(tokenizer, "conversation", None)
        if (
            conv is not None
            and hasattr(conv, "get_role_prefix")
            and hasattr(conv, "roles")
        ):
            self.instruct_user_prefix = conv.get_role_prefix(conv.roles[0])
            self.instruct_assistant_prefix = conv.get_role_prefix(conv.roles[1])
            self.instruct_user_sep = conv.sep
            self.instruct_assistant_sep = conv.sep2 or conv.sep
        else:
            self.instruct_user_prefix = "User: "
            self.instruct_assistant_prefix = "Assistant: "
            self.instruct_user_sep = "\n\n"
            self.instruct_assistant_sep = tokenizer.eos_token or "\n\n"

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

    def _append_prompt_sections(self, sections, prompt, system_prompt, uncond_kwargs, template):
        if template == "instruct":
            if system_prompt:
                sections.append(dict(type="text", text=system_prompt))
                sections.append(dict(type="text", text=self.instruct_user_sep))
            sections.append(dict(type="text", text=self.instruct_user_prefix))
            sections.append(dict(type="text", text=prompt, **uncond_kwargs))
            sections.append(dict(type="text", text=self.instruct_user_sep))
            sections.append(dict(type="text", text=self.instruct_assistant_prefix))
        else:
            if system_prompt:
                sections.append(dict(type="text", text=system_prompt))
            sections.append(dict(type="text", text=prompt, **uncond_kwargs))

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
                          <timestep> <img>*N <eoi>
                instruct: <bos> [system]\n\nUser: [user]\n\nAssistant: [cot]
                          <answer> <boi> <img_size> <img_ratio> <timestep>
                          <img>*N <eoi> </answer>
        """
        uncond_kwargs = dict(uncond_p=uncond_p)

        sections = []

        self._append_prompt_sections(
            sections, prompt, system_prompt, uncond_kwargs, template
        )
        if cot_text:
            sections.append(dict(type="text", text=cot_text, **uncond_kwargs))
        if template == "instruct" and self.answer_token_id is not None:
            sections.append(dict(type="text", text="<answer>", ignore=True))

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

        # </answer> and assistant separator after image (instruct template)
        if template == "instruct":
            if self.end_of_answer_token_id is not None:
                sections.append(dict(type="text", text="</answer>", ignore=True))
            sections.append(dict(type="text", text=self.instruct_assistant_sep))

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

    def build_ti2i_sequence(
        self,
        prompt,
        token_h,
        token_w,
        cond_image_infos,
        base_size=1024,
        ratio_idx=0,
        cot_text="",
        system_prompt="",
        uncond_p=0.0,
        template="pretrain",
    ):
        """Build the TI2I token sequence with both joint_image and gen_image sections.

        The token sequence structure:
          <bos> [system] [prompt] [cot] <boi> [cond_meta] <img>*vae_len
          <joint_img_sep> <img>*vit_len <eoi> <boi> [gen_meta] <img>*N <eoi> <eos>

        Args:
            cond_image_infos: list of dicts, each with:
                vae_token_h, vae_token_w, vit_token_h, vit_token_w,
                base_size, ratio_idx
        """
        uncond_kwargs = dict(uncond_p=uncond_p)
        sections = []

        self._append_prompt_sections(
            sections, prompt, system_prompt, uncond_kwargs, template
        )
        if cot_text:
            sections.append(dict(type="text", text=cot_text, **uncond_kwargs))
        if template == "instruct" and self.answer_token_id is not None:
            sections.append(dict(type="text", text="<answer>", ignore=True))

        # Add joint_image sections for each condition image
        for info in cond_image_infos:
            vae_len = info["vae_token_h"] * info["vae_token_w"]
            vit_len = info["vit_token_h"] * info["vit_token_w"]
            sections.append(
                dict(
                    type="joint_image",
                    token_length=[vae_len, vit_len],
                    add_timestep_token=True,
                    use_front_boi_token=True,
                    add_image_shape_token=True,
                    base_size=info.get("base_size", base_size),
                    ratio_idx=info.get("ratio_idx", ratio_idx),
                )
            )

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

        if template == "instruct":
            if self.end_of_answer_token_id is not None:
                sections.append(dict(type="text", text="</answer>", ignore=True))
            sections.append(dict(type="text", text=self.instruct_assistant_sep))

        return self._encode_sections(sections)

    def build_ti2i_uncond_sequence(
        self,
        prompt,
        token_h,
        token_w,
        cond_image_infos,
        base_size=1024,
        ratio_idx=0,
        cot_text="",
        system_prompt="",
        template="pretrain",
    ):
        """Build the unconditional TI2I token sequence for CFG."""
        return self.build_ti2i_sequence(
            prompt=prompt,
            token_h=token_h,
            token_w=token_w,
            cond_image_infos=cond_image_infos,
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
        gen_image_scatter_index = self._slices_to_index(gen_image_slices)

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
        cond_vae_image_slices, cond_vae_image_mask = self._parse_image_slices(
            extra_token_pos, "vae_img", full_seq_tensor
        )
        cond_vae_scatter_index = self._slices_to_index(cond_vae_image_slices)
        cond_vit_image_slices, cond_vit_image_mask = self._parse_image_slices(
            extra_token_pos, "vit_img", full_seq_tensor
        )
        cond_vit_scatter_index = self._slices_to_index(cond_vit_image_slices)

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
            image_scatter_index=gen_image_scatter_index,
            timestep_scatter_index=timestep_scatter_index,
            gen_timestep_scatter_index=gen_timestep_scatter_index,
            cond_timestep_scatter_index=cond_timestep_scatter_index,
            guidance_scatter_index=guidance_scatter_index,
            cond_vae_image_mask=cond_vae_image_mask,
            cond_vae_scatter_index=cond_vae_scatter_index,
            cond_vit_image_mask=cond_vit_image_mask,
            cond_vit_scatter_index=cond_vit_scatter_index,
            text_mask=text_mask,
            gen_image_slices=gen_image_slices,
            all_image_slices=all_image_slices,
        )

    def _build_token_sequence(
        self, template, token_source, add_bos=True, add_eos=False
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

    def _parse_image_slices(self, extra_token_pos, prefix, tokens):
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

    @staticmethod
    def _slices_to_index(image_slices):
        if not image_slices:
            return None
        return torch.cat(
            [
                torch.arange(image_slice.start, image_slice.stop, dtype=torch.long)
                for image_slice in image_slices
            ],
            dim=0,
        )


# ---------------------------------------------------------------------------
# HunyuanImage3BeforeDenoisingStage
# ---------------------------------------------------------------------------


class HunyuanImage3BeforeDenoisingStage(PipelineStage):
    """BeforeDenoisingStage for HunyuanImage-3.0 (Phase 1 - DiT-only T2I).

    Prepares all inputs needed by DenoisingStage:
    - Token sequences (conditional + unconditional) via HunyuanImage3Tokenizer
    - Token embeddings via transformer.embed_tokens
    - 2D RoPE (cos, sin) via build_batch_2d_rope
    - Initial noise latents
    - Timesteps via FlowMatchEulerDiscreteScheduler
    - All masks and scatter indices

    Note: AR generation (CoT / ratio prediction) is not supported in Phase 1.
    """

    def __init__(self, vae, transformer, tokenizer, scheduler, image_encoder=None):
        super().__init__()
        self.vae = vae
        self.transformer = transformer
        self.tokenizer_wrapper = HunyuanImage3Tokenizer(tokenizer)
        self.scheduler = scheduler
        self.image_encoder = image_encoder  # dict with vision_model, vision_aligner, vit_processor_config
        self.resolution_group = ResolutionGroup(
            base_size=1024,
            step=64,
            align=1,
            extra_resolutions=[
                Resolution(s) for s in HUNYUAN_IMAGE3_EXTRA_RESOLUTIONS
            ],
        )

    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]:
        stage_name = self._component_stage_name(stage_name)
        return [
            ComponentUse(
                stage_name=stage_name,
                component_name="transformer",
                memory_intensive=True,
            ),
        ]

    def _manage_transformer_use(self) -> None:
        manager = self._component_residency_manager
        if manager is None:
            return
        use = self._declared_component_use(component_name="transformer")
        manager.begin_use(use, module=self.transformer)

    def _embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.transformer.embed_tokens(tokens)

    @staticmethod
    def _build_rope_image_infos(
        tokenizer_output: TokenizerOutput,
        token_h: int,
        token_w: int,
        is_ti2i: bool,
        cond_image_infos,
    ):
        if not tokenizer_output.all_image_slices:
            return None
        if is_ti2i and cond_image_infos is not None:
            all_image_slices = tokenizer_output.all_image_slices
            rope_image_infos = []
            slice_idx = 0
            for info in cond_image_infos:
                if slice_idx < len(all_image_slices):
                    s = all_image_slices[slice_idx]
                    rope_image_infos.append(
                        (s, (info["vae_token_h"], info["vae_token_w"]))
                    )
                    slice_idx += 1

                if slice_idx < len(all_image_slices):
                    s = all_image_slices[slice_idx]
                    if (
                        info.get("vit_token_h", 0) > 0
                        and info.get("vit_token_w", 0) > 0
                    ):
                        rope_image_infos.append(
                            (s, (info["vit_token_h"], info["vit_token_w"]))
                        )
                    slice_idx += 1

            if slice_idx < len(all_image_slices):
                s = all_image_slices[slice_idx]
                rope_image_infos.append((s, (token_h, token_w)))
            return [rope_image_infos]
        return [[(s, (token_h, token_w)) for s in tokenizer_output.all_image_slices]]

    @staticmethod
    def _build_encoder_attention_mask(
        seq_len: int,
        image_slices: list,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        attention_mask = torch.ones(
            seq_len, seq_len, device=device, dtype=torch.bool
        ).tril(diagonal=0)
        for img_slice in image_slices or []:
            attention_mask[img_slice, img_slice] = True
        return attention_mask.unsqueeze(0).unsqueeze(0)

    @staticmethod
    def _batch_to_device(value: torch.Tensor | None, device: torch.device):
        return value.unsqueeze(0).to(device) if value is not None else None

    @staticmethod
    def _normalize_bot_task(bot_task):
        bot_task = bot_task or ""
        if bot_task in ("none", "vanilla"):
            return ""
        return bot_task

    @staticmethod
    def _is_auto_image_size(image_size):
        return image_size is None or str(image_size).strip().lower() == "auto"

    def _parse_requested_image_size(self, image_size):
        """Parse official HunyuanImage3 image_size formats.

        Returns (width, height, ratio_index).  For auto mode, all values are
        None so the AR ratio predictor can choose the bucket.
        """
        if self._is_auto_image_size(image_size):
            return None, None, None

        if isinstance(image_size, str):
            value = image_size.strip()
            if value.startswith("<img_ratio_") and value.endswith(">"):
                ratio_index = int(value[len("<img_ratio_") : -1])
                reso = self.resolution_group[ratio_index]
                return reso.w, reso.h, ratio_index
            if "x" in value:
                height, width = [int(part) for part in value.lower().split("x", 1)]
                return width, height, None
            if ":" in value:
                width, height = [int(part) for part in value.split(":", 1)]
                return width, height, None
            raise ValueError(
                "`image_size` should be 'auto', 'HxW', 'W:H', or '<img_ratio_i>', "
                f"got {image_size!r}."
            )

        if isinstance(image_size, (list, tuple)) and len(image_size) == 2:
            height, width = [int(part) for part in image_size]
            return width, height, None

        raise ValueError(
            "`image_size` should be 'auto', 'HxW', 'W:H', '<img_ratio_i>', "
            f"or a 2-item sequence, got {image_size!r}."
        )

    def _build_tokenizer_outputs(
        self,
        *,
        prompt,
        token_h,
        token_w,
        cond_image_infos,
        base_size,
        ratio_idx,
        cot_text,
        system_prompt,
        template,
        do_cfg,
    ):
        common_kwargs = dict(
            prompt=prompt,
            token_h=token_h,
            token_w=token_w,
            base_size=base_size,
            ratio_idx=ratio_idx,
            cot_text=cot_text,
            system_prompt=system_prompt,
            template=template,
        )
        if cond_image_infos is not None:
            cond_output = self.tokenizer_wrapper.build_ti2i_sequence(
                cond_image_infos=cond_image_infos,
                uncond_p=0.0,
                **common_kwargs,
            )
            uncond_output = (
                self.tokenizer_wrapper.build_ti2i_uncond_sequence(
                    cond_image_infos=cond_image_infos,
                    **common_kwargs,
                )
                if do_cfg
                else None
            )
        else:
            cond_output = self.tokenizer_wrapper.build_t2i_sequence(
                uncond_p=0.0,
                **common_kwargs,
            )
            uncond_output = (
                self.tokenizer_wrapper.build_uncond_sequence(**common_kwargs)
                if do_cfg
                else None
            )
        return cond_output, uncond_output

    # ------------------------------------------------------------------
    # TI2I: Condition image processing
    # ------------------------------------------------------------------

    @staticmethod
    def _resize_and_crop_center(image, target_width, target_height):
        """Resize and center-crop a PIL image to target dimensions.

        Mirrors the official HunyuanImage3Processor._resize_and_crop.
        """
        tw, th = target_width, target_height
        w, h = image.size
        tr = th / tw
        r = h / w
        if r < tr:
            resize_height = th
            resize_width = int(round(th / h * w))
        else:
            resize_width = tw
            resize_height = int(round(tw / w * h))
        import torchvision.transforms.functional as TF

        resized = TF.resize(image, (resize_height, resize_width))
        crop_top = int(round((resize_height - th) / 2.0))
        crop_left = int(round((resize_width - tw) / 2.0))
        return TF.crop(resized, crop_top, crop_left, th, tw)

    def _process_cond_images(self, batch, device):
        """Process condition images for TI2I mode.

        For each condition image:
        - VAE branch: resize+crop -> VAE encode -> latent + timestep=0
        - ViT branch: original image -> SigLIP2 + LightProjector -> ViT embeddings

        Returns:
            (cond_vae_images, cond_timestep, cond_vit_embeds, cond_image_info)
            where cond_image_info is a dict with token dimensions for sequence building.
        """
        from sglang.multimodal_gen.runtime.models.vision_utils import load_image

        # Load condition images
        image_path = getattr(batch, "image_path", None)
        if image_path is None:
            return None, None, None, None

        if isinstance(image_path, list):
            pil_images = [load_image(p) for p in image_path]
        else:
            pil_images = [load_image(image_path)]

        import torchvision.transforms as T

        vae_transform = T.Compose([T.ToTensor(), T.Normalize([0.5], [0.5])])

        vision_model = self.image_encoder["vision_model"]
        vision_aligner = self.image_encoder["vision_aligner"]
        vit_processor = self.image_encoder.get("vit_processor")

        cond_vae_image_list = []
        cond_t_list = []
        cond_vit_embeds_list = []
        cond_image_info_list = []

        for pil_image in pil_images:
            orig_width, orig_height = pil_image.size

            # Get target resolution for VAE
            target_w, target_h = self.resolution_group.get_target_size(
                orig_width, orig_height
            )
            target_w, target_h = int(target_w), int(target_h)

            base_size, ratio_idx = self.resolution_group.get_base_size_and_ratio_index(
                target_w, target_h
            )

            # --- VAE branch ---
            vae_image = self._resize_and_crop_center(pil_image, target_w, target_h)
            vae_tensor = vae_transform(vae_image).unsqueeze(0).unsqueeze(2)  # [1, C, 1, H, W]
            vae_tensor = vae_tensor.to(device)

            vae_config = self.vae.config
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=True):
                vae_result = self.vae.encode(vae_tensor)
                if isinstance(vae_result, torch.Tensor):
                    vae_latent = vae_result
                else:
                    _gen = torch.Generator(device=device).manual_seed(0)
                    vae_latent = vae_result.latent_dist.sample(_gen)
                if hasattr(vae_config, "shift_factor") and vae_config.shift_factor:
                    vae_latent = vae_latent.sub_(vae_config.shift_factor)
                if hasattr(vae_config, "scaling_factor") and vae_config.scaling_factor:
                    vae_latent = vae_latent.mul_(vae_config.scaling_factor)

            # Squeeze temporal dimension if present
            if vae_latent.ndim == 5 and vae_latent.shape[2] == 1:
                vae_latent = vae_latent.squeeze(2)  # [1, C, H, W]

            # Timestep=0 for clean conditional image
            cond_t = torch.zeros((1,), device=device)

            cond_vae_image_list.append(vae_latent.squeeze(0))  # [C, H, W]
            cond_t_list.append(cond_t)

            # --- ViT branch ---
            vit_embeds = None
            vit_token_h = 0
            vit_token_w = 0
            if vit_processor is not None and vision_model is not None:
                vit_inputs = vit_processor(pil_image, return_tensors="pt")
                pixel_values = vit_inputs["pixel_values"].to(device)
                spatial_shapes = vit_inputs["spatial_shapes"].squeeze(0).to(device)
                pixel_attention_mask = vit_inputs["pixel_attention_mask"].squeeze(0).to(device)

                vit_token_h = int(spatial_shapes[0].item())
                vit_token_w = int(spatial_shapes[1].item())

                # Run vision encoder + aligner
                vision_output = vision_model(
                    pixel_values,
                    attention_mask=pixel_attention_mask.unsqueeze(0),
                    spatial_shapes=spatial_shapes.unsqueeze(0),
                )
                image_embed = vision_output.last_hidden_state
                if vision_aligner is not None:
                    image_embed = vision_aligner(image_embed)

                # Flatten: [1, num_patches, dim] -> [num_patches * dim]
                vit_embeds = image_embed.reshape(-1)
                cond_vit_embeds_list.append(vit_embeds)

            # Compute token dimensions for VAE
            vae_downsample = 16  # default VAE downsample factor
            vae_token_h = target_h // vae_downsample
            vae_token_w = target_w // vae_downsample

            cond_image_info_list.append({
                "vae_token_h": vae_token_h,
                "vae_token_w": vae_token_w,
                "vit_token_h": vit_token_h,
                "vit_token_w": vit_token_w,
                "base_size": int(base_size),
                "ratio_idx": int(ratio_idx),
                "target_width": target_w,
                "target_height": target_h,
            })

        # Stack VAE images into batch tensor
        if len(cond_vae_image_list) == 1:
            cond_vae_images = cond_vae_image_list[0].unsqueeze(0)  # [1, C, H, W]
            cond_timestep = cond_t_list[0]  # [1]
        else:
            cond_vae_images = torch.stack(cond_vae_image_list, dim=0)
            cond_timestep = torch.cat(cond_t_list, dim=0)

        # Concatenate ViT embeddings
        if cond_vit_embeds_list:
            cond_vit_embeds = torch.cat(cond_vit_embeds_list, dim=0)
        else:
            cond_vit_embeds = None

        return cond_vae_images, cond_timestep, cond_vit_embeds, cond_image_info_list

    # ------------------------------------------------------------------
    # Main forward
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        """Process batch and populate all fields needed by DenoisingStage."""
        self._manage_transformer_use()
        device = get_local_torch_device()
        dtype = torch.bfloat16

        batch.bot_task = self._normalize_bot_task(getattr(batch, "bot_task", ""))

        # Phase 1 validation: AR generation is not supported
        if batch.bot_task:
            raise ValueError(
                "AR generation is not supported in this version. Please leave bot_task empty."
            )

        image_size = getattr(batch, "image_size", "auto") or "auto"

        # Phase 1 validation: auto image size requires AR generation
        if self._is_auto_image_size(image_size):
            raise ValueError(
                "Auto image size requires AR generation (not supported in this version). "
                "Please specify explicit image_size like '1024x1024'."
            )

        requested_width, requested_height, requested_ratio_index = (
            self._parse_requested_image_size(image_size)
        )
        if requested_ratio_index is not None:
            batch.ratio_index = requested_ratio_index

        # 1. Parse inputs from batch
        prompt = _first_prompt(batch.prompt)
        cot_text = ""  # Phase 1: no AR generation, cot_text is always empty
        bot_task = getattr(batch, "bot_task", "") or ""
        system_prompt = ""  # Phase 1: system prompt is always empty
        template = getattr(batch, "sequence_template", "pretrain") or "pretrain"

        height = batch.height
        width = batch.width
        if requested_width is not None and requested_height is not None:
            width = requested_width
            height = requested_height
        guidance_scale = batch.guidance_scale
        num_inference_steps = batch.num_inference_steps
        seed = batch.seed

        # 1b. If ratio_index was specified via image_size, override height/width
        if hasattr(batch, "ratio_index") and batch.ratio_index is not None:
            reso = self.resolution_group[batch.ratio_index]
            height = reso.h
            width = reso.w

        # 2. Get target resolution from bucket
        target_w, target_h = self.resolution_group.get_target_size(width, height)
        batch.width = int(target_w)
        batch.height = int(target_h)

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
        # Detect TI2I mode: image_path is set
        is_ti2i = getattr(batch, "image_path", None) is not None

        cond_vae_images = None
        cond_timestep = None
        cond_vit_embeds = None
        cond_image_infos = None

        if is_ti2i:
            # Process condition images (VAE encode + ViT encode)
            (
                cond_vae_images,
                cond_timestep,
                cond_vit_embeds,
                cond_image_infos,
            ) = self._process_cond_images(batch, device)

        # 6. Build conditional/unconditional token sequences.
        do_cfg = guidance_scale > 1.0
        cond_output, uncond_output = self._build_tokenizer_outputs(
            prompt=prompt,
            token_h=token_h,
            token_w=token_w,
            cond_image_infos=cond_image_infos if is_ti2i else None,
            base_size=base_size,
            ratio_idx=ratio_idx,
            cot_text=cot_text,
            system_prompt=system_prompt,
            template=template,
            do_cfg=do_cfg,
        )

        # 7. Get token embeddings via transformer.embed_tokens
        cond_tokens = cond_output.tokens.unsqueeze(0).to(device)  # [1, seq_len]
        cond_embeds = self._embed_tokens(cond_tokens)  # [1, seq_len, hidden_size]

        if do_cfg:
            assert uncond_output is not None
            uncond_tokens = uncond_output.tokens.unsqueeze(0).to(device)
            uncond_embeds = self._embed_tokens(uncond_tokens)

        # 8. Build 2D RoPE
        seq_len = cond_tokens.shape[1]
        n_elem = dit_config.rope_axes_dim[0] + dit_config.rope_axes_dim[1]  # 128
        rope_cos, rope_sin = build_batch_2d_rope(
            seq_len=seq_len,
            n_elem=n_elem,
            image_infos=self._build_rope_image_infos(
                cond_output,
                token_h,
                token_w,
                is_ti2i,
                cond_image_infos,
            ),
            device=device,
            base=dit_config.rope_theta,
        )
        rope_2d = (rope_cos, rope_sin)
        if do_cfg:
            neg_seq_len = uncond_tokens.shape[1]
            neg_rope_cos, neg_rope_sin = build_batch_2d_rope(
                seq_len=neg_seq_len,
                n_elem=n_elem,
                image_infos=self._build_rope_image_infos(
                    uncond_output,
                    token_h,
                    token_w,
                    is_ti2i,
                    cond_image_infos,
                ),
                device=device,
                base=dit_config.rope_theta,
            )
            neg_rope_2d = (neg_rope_cos, neg_rope_sin)

        # 9. Build position IDs
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)  # [1, seq_len]
        if do_cfg:
            neg_position_ids = torch.arange(neg_seq_len, device=device).unsqueeze(0)

        # 10. Build hybrid attention mask (causal for text, bidirectional for image)
        # Official approach: start with lower-triangular causal mask, then set
        # image token positions to True (bidirectional within image regions).
        encoder_attention_mask = self._build_encoder_attention_mask(
            seq_len, cond_output.all_image_slices, device, dtype
        )
        if do_cfg:
            neg_encoder_attention_mask = self._build_encoder_attention_mask(
                neg_seq_len, uncond_output.all_image_slices, device, dtype
            )

        # 11. Prepare image masks and scatter indices
        image_mask = self._batch_to_device(cond_output.image_mask, device)
        image_scatter_index = self._batch_to_device(
            cond_output.image_scatter_index, device
        )
        if do_cfg:
            neg_image_mask = self._batch_to_device(uncond_output.image_mask, device)
            neg_image_scatter_index = self._batch_to_device(
                uncond_output.image_scatter_index, device
            )
        timestep_scatter_index = self._batch_to_device(
            cond_output.timestep_scatter_index, device
        )
        if do_cfg:
            neg_timestep_scatter_index = self._batch_to_device(
                uncond_output.timestep_scatter_index, device
            )
        gen_timestep_scatter_index = self._batch_to_device(
            cond_output.gen_timestep_scatter_index, device
        )
        if do_cfg:
            neg_gen_timestep_scatter_index = self._batch_to_device(
                uncond_output.gen_timestep_scatter_index, device
            )
        cond_timestep_scatter_index = self._batch_to_device(
            cond_output.cond_timestep_scatter_index, device
        )
        cond_vae_image_mask = self._batch_to_device(
            cond_output.cond_vae_image_mask, device
        )
        cond_vae_scatter_index = self._batch_to_device(
            cond_output.cond_vae_scatter_index, device
        )
        cond_vit_image_mask = self._batch_to_device(
            cond_output.cond_vit_image_mask, device
        )
        cond_vit_scatter_index = self._batch_to_device(
            cond_output.cond_vit_scatter_index, device
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
        batch.image_scatter_index = image_scatter_index
        if do_cfg:
            batch.neg_image_mask = neg_image_mask
            batch.neg_image_scatter_index = neg_image_scatter_index
        batch.timestep_scatter_index = timestep_scatter_index
        if do_cfg:
            batch.neg_timestep_scatter_index = neg_timestep_scatter_index
        batch.gen_timestep_scatter_index = gen_timestep_scatter_index
        if do_cfg:
            batch.neg_gen_timestep_scatter_index = neg_gen_timestep_scatter_index
        batch.cond_timestep_scatter_index = cond_timestep_scatter_index
        batch.cond_vae_image_mask = cond_vae_image_mask
        batch.cond_vae_scatter_index = cond_vae_scatter_index
        batch.cond_vit_image_mask = cond_vit_image_mask
        batch.cond_vit_scatter_index = cond_vit_scatter_index
        batch.cond_vit_embeds = cond_vit_embeds  # ViT embeddings for TI2I
        # TI2I condition fields
        if cond_vae_images is not None:
            batch.cond_vae_images = cond_vae_images
        if cond_timestep is not None:
            batch.cond_timestep = cond_timestep
        batch.rope_2d = rope_2d
        if do_cfg:
            batch.neg_rope_2d = neg_rope_2d
        batch.position_ids = position_ids
        if do_cfg:
            batch.neg_position_ids = neg_position_ids
        batch.token_h = token_h
        batch.token_w = token_w

        batch.height = target_h
        batch.width = target_w
        batch.raw_latent_shape = latents.shape

        return batch