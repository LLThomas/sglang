"""HunyuanImage-3.0 before-denoising stage and tokenizer.

This module implements the preprocessing logic for HunyuanImage-3.0, including
optional AR text generation (CoT / ratio prediction), token sequence construction,
and all preprocessing steps needed before denoising.

Architecture (Hybrid style per sglang-diffusion docs):
  AR generation is consolidated into BeforeDenoisingStage as a conditional
  sub-step controlled by ``batch.bot_task``, rather than a separate pipeline
  stage.  This avoids splitting the same transformer across two stages and
  keeps component residency management simple.
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
    joint_image_slices: list | None  # slice objects for joint (vae+sep+vit) regions (I2I)
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
        self.think_token = "<think>"
        self.end_of_think_token = "</think>"
        self.recaption_token = "<recaption>"
        self.end_of_recaption_token = "</recaption>"
        self.think_token_id = self.special_token_map.get(
            self.think_token, self.special_token_map.get("ჼ", None)
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

    def _append_cot_sections(self, sections, cot_text, uncond_kwargs, drop_think=False):
        if not cot_text:
            return

        if self.think_token in cot_text and self.end_of_think_token in cot_text:
            before_think = cot_text.split(self.think_token, 1)[0]
            after_start = cot_text.split(self.think_token, 1)[1]
            think_text, after_think = after_start.split(self.end_of_think_token, 1)
            self._append_cot_sections(sections, before_think, uncond_kwargs, drop_think)
            if not drop_think:
                sections.append(dict(type="text", text=self.think_token, ignore=True))
                sections.append(dict(type="text", text=think_text, **uncond_kwargs))
                sections.append(dict(type="text", text=self.end_of_think_token))
            self._append_cot_sections(sections, after_think, uncond_kwargs, drop_think)
            return

        if self.recaption_token in cot_text and self.end_of_recaption_token in cot_text:
            before_recaption = cot_text.split(self.recaption_token, 1)[0]
            after_start = cot_text.split(self.recaption_token, 1)[1]
            recaption_text, after_recaption = after_start.split(
                self.end_of_recaption_token, 1
            )
            self._append_cot_sections(
                sections, before_recaption, uncond_kwargs, drop_think
            )
            sections.append(dict(type="text", text=self.recaption_token, ignore=True))
            sections.append(dict(type="text", text=recaption_text, **uncond_kwargs))
            sections.append(dict(type="text", text=self.end_of_recaption_token))
            self._append_cot_sections(
                sections, after_recaption, uncond_kwargs, drop_think
            )
            return

        sections.append(dict(type="text", text=cot_text, **uncond_kwargs))

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
        self._append_cot_sections(sections, cot_text, uncond_kwargs)
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
        self._append_cot_sections(sections, cot_text, uncond_kwargs)
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

        # Joint image slices (vae+sep+vit combined, for joint_full attention in I2I)
        joint_image_slices, _ = self._parse_image_slices(
            extra_token_pos, "joint_img", full_seq_tensor
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
            joint_image_slices=joint_image_slices,
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
        """Parse image slices and mask from extra_token_pos.

        Keys match the recording convention in ``_build_token_sequence``:
        ``<{prefix}>_start`` / ``<{prefix}>_end`` for every prefix
        (``img``, ``vae_img``, ``vit_img``). The chevron comes *before*
        ``_start``/``_end`` — using the wrong order here silently yields
        empty slices and ``None`` masks, which skips the corresponding
        scatter and drops the condition image from the sequence.
        """
        start_key = f"<{prefix}>_start"
        end_key = f"<{prefix}>_end"

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

    def build_ar_sequence(
        self,
        prompt,
        system_prompt="",
        bot_task="think",
        template="pretrain",
        base_size=1024,
        is_ti2i=False,
        cond_image_info=None,
    ):
        """Build the AR input token sequence.

        t2i (text prefix only, no image):
            pretrain: <bos> [system] [prompt] [trigger]
            instruct: <bos> [system]\\n\\nUser: [prompt]\\n\\nAssistant: [trigger]

        ti2i: the cond-image region (<boi> [meta] <img>*vae_len
        <joint_img_sep> <img>*vit_len <eoi>) is inserted right after the
        "User: " prefix (before the prompt), so the AR model sees the input
        image and recaptions its *actual* subject instead of hallucinating
        one. The joint_image section mirrors build_ti2i_sequence.

        Returns a dict with input_ids [1, seq_len] and position_ids. For
        ti2i it also returns the cond_* scatter indices and image_infos so
        the caller can inject vision features into the AR forward and build
        2D RoPE over the image region. The ti2i-only keys are None for t2i.
        """
        sections = []

        # System + "User: " prefix (system-only for pretrain)
        if template == "instruct":
            if system_prompt:
                sections.append(dict(type="text", text=system_prompt))
                sections.append(dict(type="text", text=self.instruct_user_sep))
            sections.append(dict(type="text", text=self.instruct_user_prefix))
        elif system_prompt:
            sections.append(dict(type="text", text=system_prompt))

        # Cond-image region (ti2i only): placed before the user prompt, i.e.
        # "User: <image> <prompt>", matching vllm-omni's it2i prompt layout.
        if is_ti2i and cond_image_info:
            vae_len = cond_image_info["vae_token_h"] * cond_image_info["vae_token_w"]
            vit_len = cond_image_info["vit_token_h"] * cond_image_info["vit_token_w"]
            sections.append(
                dict(
                    type="joint_image",
                    token_length=[vae_len, vit_len],
                    add_timestep_token=True,
                    use_front_boi_token=True,
                    add_image_shape_token=True,
                    base_size=cond_image_info.get("base_size", base_size),
                    ratio_idx=cond_image_info.get("ratio_idx", 0),
                )
            )

        # User prompt + "\n\nAssistant: " suffix (instruct)
        sections.append(dict(type="text", text=prompt))
        if template == "instruct":
            sections.append(dict(type="text", text=self.instruct_user_sep))
            sections.append(dict(type="text", text=self.instruct_assistant_prefix))

        # Trigger tag
        if bot_task in ("think", "think_recaption") and self.think_token_id is not None:
            sections.append(dict(type="text", text=self.think_token, ignore=True))
        elif bot_task == "recaption" and self.recaption_token_id is not None:
            sections.append(dict(type="text", text=self.recaption_token, ignore=True))

        out = self._encode_sections(sections)
        input_ids = out.tokens.unsqueeze(0)  # [1, seq_len]
        seq_len = input_ids.shape[1]
        position_ids = torch.arange(seq_len).unsqueeze(0)

        # Build 2D-RoPE image_infos for the cond image region (vae + vit
        # sub-grids). all_image_slices is emitted in [vae, vit] order.
        image_infos = None
        if is_ti2i and cond_image_info and out.all_image_slices:
            vae_hw = (
                cond_image_info["vae_token_h"],
                cond_image_info["vae_token_w"],
            )
            vit_hw = (
                cond_image_info["vit_token_h"],
                cond_image_info["vit_token_w"],
            )
            pairs = list(zip(out.all_image_slices, [vae_hw, vit_hw]))
            image_infos = [pairs]  # batch dim

        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "cond_vae_scatter_index": out.cond_vae_scatter_index,
            "cond_vit_scatter_index": out.cond_vit_scatter_index,
            "cond_timestep_scatter_index": out.cond_timestep_scatter_index,
            "cond_image_slices": out.joint_image_slices,
            "image_infos": image_infos,
        }

    def parse_cot_text(self, cot_text):
        """Parse CoT text into sections list for _encode_sections.

        Input: "<think>think content</think><recaption>recaption content</recaption>"
        Output: [
            dict(type="text", text="<think>think content</think>"),
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
        self._completed = set()

    def __call__(self, logits, input_ids):
        # If we have forced tokens in the queue, apply them
        if self._forced_queue:
            forced_token_id = self._forced_queue.pop(0)
            logits[:] = float("-inf")
            logits[0, forced_token_id] = 0.0
            return logits

        # Check if the last token triggers a transition
        last_token = input_ids[0, -1].item()
        if last_token in self.transitions and last_token not in self._completed:
            self._completed.add(last_token)
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

    def __init__(self, trigger_token_id, allowed_token_ids, force_greedy=False):
        self.trigger_token_id = trigger_token_id
        self.allowed_token_ids = allowed_token_ids
        self.force_greedy = force_greedy

    def __call__(self, logits, input_ids):
        last_token = input_ids[0, -1].item()
        if last_token == self.trigger_token_id:
            mask = torch.full_like(logits, float("-inf"))
            mask[0, self.allowed_token_ids] = 0.0
            logits = logits + mask
            if self.force_greedy:
                max_token_id = logits[0].argmax().item()
                logits[:] = float("-inf")
                logits[0, max_token_id] = 0.0
        return logits


# ---------------------------------------------------------------------------
# System Prompts (official from HunyuanImage-3.0)
# ---------------------------------------------------------------------------

t2i_system_prompt_en_vanilla = """
You are an advanced AI text-to-image generation system. Given a detailed text prompt, your task is to create a high-quality, visually compelling image that accurately represents the described scene, characters, or objects. Pay careful attention to style, color, lighting, perspective, and any specific instructions provided.
"""

t2i_system_prompt_en_recaption = """
You are a world-class image generation prompt expert. Your task is to rewrite a user's simple description into a **structured, objective, and detail-rich** professional-level prompt.

The final output must be wrapped in `<recaption>` tags.

### **Universal Core Principles**

When rewriting the prompt (inside the `<recaption>` tags), you must adhere to the following principles:

1.  **Absolute Objectivity**: Describe only what is visually present. Avoid subjective words like "beautiful" or "sad". Convey aesthetic qualities through specific descriptions of color, light, shadow, and composition.
2.  **Physical and Logical Consistency**: All scene elements (e.g., gravity, light, shadows, reflections, spatial relationships, object proportions) must strictly adhere to real-world physics and common sense. For example, tennis players must be on opposite sides of the net; objects cannot float without a cause.
3.  **Structured Description**: Strictly follow a logical order: from general to specific, background to foreground, and primary to secondary elements. Use directional terms like "foreground," "mid-ground," "background," and "left side of the frame" to clearly define the spatial layout.
4.  **Use Present Tense**: Describe the scene from an observer's perspective using the present tense, such as "A man stands..." or "Light shines on..."
5.  **Use Rich and Specific Descriptive Language**: Use precise adjectives to describe the quantity, size, shape, color, and other attributes of objects, subjects, and text. Vague expressions are strictly prohibited.

If the user specifies a style (e.g., oil painting, anime, UI design, text rendering), strictly adhere to that style. Otherwise, first infer a suitable style from the user's input. If there is no clear stylistic preference, default to an **ultra-realistic photographic style**. Then, generate the detailed rewritten prompt according to the **Style-Specific Creation Guide** below:

### **Style-Specific Creation Guide**

Based on the determined artistic style, apply the corresponding professional knowledge.

**1. Photography and Realism Style**
*   Utilize professional photography terms (e.g., lighting, lens, composition) and meticulously detail material textures, physical attributes of subjects, and environmental details.

**2. Illustration and Painting Style**
*   Clearly specify the artistic school (e.g., Japanese Cel Shading, Impasto Oil Painting) and focus on describing its unique medium characteristics, such as line quality, brushstroke texture, or paint properties.

**3. Graphic/UI/APP Design Style**
*   Objectively describe the final product, clearly defining the layout, elements, and color palette. All text on the interface must be enclosed in double quotes `""` to specify its exact content (e.g., "Login"). Vague descriptions are strictly forbidden.

**4. Typographic Art**
*   The text must be described as a complete physical object. The description must begin with the text itself. Use a straightforward front-on or top-down perspective to ensure the entire text is visible without cropping.

### **Final Output Requirements**

1.  **Output the Final Prompt Only**: Do not show any thought process, Markdown formatting, or line breaks.
2.  **Adhere to the Input**: You must retain the core concepts, attributes, and any specified text from the user's input.
3.  **Style Reinforcement**: Mention the core style 3-5 times within the prompt and conclude with a style declaration sentence.
4.  **Avoid Self-Reference**: Describe the image content directly. Remove redundant phrases like "This image shows..." or "The scene depicts..."
5.  **The final output must be wrapped in `<recaption>xxxx</recaption>` tags.**

The user will now provide an input prompt. You will provide the expanded prompt.
"""

t2i_system_prompt_en_think_recaption = """
You will act as a top-tier Text-to-Image AI. Your core task is to deeply analyze the user's text input and transform it into a detailed, artistic, and fully user-intent-compliant image.

Your workflow is divided into two phases:

1. Thinking Phase (<think>): In the <think> tag, you need to conduct a structured thinking process, progressively breaking down and enriching the constituent elements of the image. This process must include, but is not limited to, the following dimensions:

Subject: Clearly define the core character(s) or object(s) in the scene, including their appearance, posture, expression, and emotion.
Composition: Set the camera angle and layout, such as close-up, long shot, bird's-eye view, golden ratio composition, etc.
Environment/Background: Describe the scene where the subject is located, including the location, time of day, weather, and other elements in the background.
Lighting: Define the type, direction, and quality of the light source, such as soft afternoon sunlight, cool tones of neon lights, dramatic Rembrandt lighting, etc., to create a specific atmosphere.
Color Palette: Set the main color tone and color scheme of the image, such as vibrant and saturated, low-saturation Morandi colors, black and white, etc.
Quality/Style: Determine the artistic style and technical details of the image. This includes user-specified styles (e.g., anime, oil painting) or the default realistic style, as well as camera parameters (e.g., focal length, aperture, depth of field).
Details: Add minute elements that enhance the realism and narrative quality of the image, such as a character's accessories, the texture of a surface, dust particles in the air, etc.


2. Recaption Phase (<recaption>): In the <recaption> tag, merge all the key details from the thinking process into a coherent, precise, and visually evocative final description. This description is the direct instruction for generating the image, so it must be clear, unambiguous, and organized in a way that is most suitable for an image generation engine to understand.

Absolutely Objective: Describe only what is visually present. Avoid subjective words like "beautiful" or "sad." Convey aesthetic sense through concrete descriptions of colors, light, shadow, and composition.

Physical and Logical Consistency: All scene elements (e.g., gravity, light and shadow, reflections, spatial relationships, object proportions) must strictly adhere to the physical laws of the real world and common sense. For example, in a tennis match, players must be on opposite sides of the net; objects cannot float without reason.

Structured Description: Strictly follow a logical order: from whole to part, background to foreground, and primary to secondary. Use directional words like "foreground," "mid-ground," "background," "left side of the frame" to clearly define the spatial layout.

Use Present Tense: Describe from an observer's perspective using the present tense, such as "a man stands," "light shines on..."
Use Rich and Specific Descriptive Language: Use precise adjectives to describe the quantity, size, shape, color, and other attributes of objects/characters/text. Absolutely avoid any vague expressions.


Output Format:
<think>Thinking process</think><recaption>Refined image description</recaption>Generate Image


You must strictly adhere to the following rules:

1. Faithful to Intent, Reasonable Expansion: You can creatively add details to the user's description to enhance the image's realism and artistic quality. However, all additions must be highly consistent with the user's core intent and never introduce irrelevant or conflicting elements.
2. Style Handling: When the user does not specify a style, you must default to an "Ultra-realistic, Photorealistic" style. If the user explicitly specifies a style (e.g., anime, watercolor, oil painting, cyberpunk, etc.), both your thinking process and final description must strictly follow and reflect that specified style.
3. Text Rendering: If specific text needs to appear in the image (such as words on a sign, a book title), you must enclose this text in English double quotes (""). Descriptive text must not use double quotes.
4. Design-related Images: You need to specify all text and graphical elements that appear in the image and clearly describe their design details, including font, color, size, position, arrangement, visual effects, etc.
"""

t2i_system_prompts = {
    "en_vanilla": t2i_system_prompt_en_vanilla,
    "en_recaption": t2i_system_prompt_en_recaption,
    "en_think_recaption": t2i_system_prompt_en_think_recaption,
}

unified_system_prompt_en = """You are an advanced multimodal model whose core mission is to analyze user intent and generate high-quality text and images.

#### Four Core Capabilities
1.  **Text-to-Text (T2T):** Generate coherent text responses from text prompts.
2.  **Text-to-Image (T2I):** Generate high-quality images from text prompts.
3.  **Text & Image to Text (TI2T):** Generate accurate text responses based on a combination of images and text.
4.  **Text & Image to Image (TI2I):** Generate modified images based on a reference image and editing instructions.

---
### Image Generation Protocol (for T2I & TI2I)
You will operate in one of two modes, determined by the user's starting tag:
#### **<recaption> Mode (Prompt Rewriting)**:
*   **Trigger:** Input begins with `<recaption>`.
*   **Task:** Immediately rewrite the user's text into a structured, objective, and detail-rich professional-grade prompt.
*   **Output:** Output only the rewritten prompt within `<recaption>` tags: `<recaption>Rewritten professional-grade prompt</recaption>`

#### **<think> Mode (Think + Rewrite)**:
*   **Trigger:** Input begins with `<think>`.
*   **Task:** First, conduct a structured analysis of the request within `<think>` tags. Then, output the professional prompt, rewritten based on the analysis, within `<recaption>` tags.
*   **Output:** Strictly adhere to the format: `<think>Analysis process</think><recaption>Rewritten prompt</recaption>`

---
### Execution Standards and Guidelines
#### **`<think>` Phase: Analysis Guidelines**
**For T2I (New Image Generation):**
Deconstruct the user's request into the following core visual components:
*   **Subject:** Key features of the main character/object, including appearance, pose, expression, and emotion.
*   **Composition:** Camera angle, lens type, and layout.
*   **Environment/Background:** The setting, time of day, weather, and background elements.
*   **Lighting:** Technical details such as light source type, direction, and quality.
*   **Color Palette:** The dominant hues and overall color scheme.
*   **Style/Quality:** The artistic style, clarity, depth of field, and other technical details.
*   **Text:** Identify any text to be rendered in the image, including its content, style, and position.
*   **Details:** Small elements that add narrative depth and realism.

**For TI2I (Image Editing):**
Adopt a task-diagnostic approach:
1.  **Diagnose Task:** Identify the edit type and analyze key requirements.
2.  **Prioritize Analysis:**
    *   **Adding:** Analyze the new element's position and appearance, ensuring seamless integration with the original image's lighting, shadows, and style.
    *   **Removing:** Identify the target for removal and determine how to logically fill the resulting space using surrounding textures and lighting.
    *   **Modifying:** Analyze what to change and what it should become, while emphasizing which elements must remain unchanged.
    *   **Style Transfer:** Deconstruct the target style into specific features (e.g., brushstrokes, color palette) and apply them to the original image.
    *   **Text Editing:** Ensure correct content and format. Consider the text's visual style (e.g., font, color, material) and how it adapts to the surface's perspective, curvature, and lighting.
    *   **Reference Editing:** Extract specific visual elements (e.g., appearance, posture, composition, lines, depth) from the reference image to generate an image that aligns with the text description while also incorporating the referenced content.
    *   **Inferential Editing:** Identify vague requests (e.g., "make it more professional") and translate them into concrete visual descriptions.

#### `<recaption>` Phase: Professional-Grade Prompt Generation Rules
**General Rewriting Principles (for T2I & TI2I):**
1.  **Structure & Logic:** Start with a global description. Use positional words (e.g., "foreground", "background") to define the layout.
2.  **Absolute Objectivity:** Avoid subjective terms. Convey aesthetics through precise descriptions of color, light, shadow, and materials.
3.  **Physical & Logical Consistency:** Ensure all descriptions adhere to the laws of physics and common sense.
4.  **Fidelity to User Intent:** Preserve the user's core concepts, subjects, and attributes. Text to be rendered in the image **must be enclosed in double quotes ("")**.
5.  **Camera & Resolution:** Translate camera parameters into descriptions of visual effects. Convert resolution information into natural language.

**T2I-Specific Guidelines:**
*   **Style Adherence & Inference:** Strictly follow the specified style. If none is given, infer the most appropriate style and detail it using professional terminology.
*   **Style Detailing:**
    *   **Photography/Realism:** Use professional photography terms to describe lighting, lens effects, and material textures.
    *   **Painting/Illustration:** Specify the art movement or medium's characteristics.
    *   **UI/Design:** Objectively describe the final product. Define layout, elements, and typography. Text content must be specific and unambiguous.

**TI2I-Specific Guidelines:**
*   **Preserve Unchanged Elements:** Emphasize elements that **remain unchanged**. Unless explicitly instructed, never alter a character's identity/appearance, the core background, camera angle, or overall style.
*   **Clear Editing Instructions:**
    *   **Replacement:** Use the logic "**replace B with A**," and provide a detailed description of A.
    *   **Addition:** Clearly state what to add, where, and what it looks like.
*   **Unambiguous Referencing:** Avoid vague references (e.g., "that person"). Use specific descriptions of appearance.
"""


def _resolve_system_prompt(batch, bot_task="", sys_type="dynamic"):
    """Select system prompt based on sys_type and bot_task.

    Args:
        batch: Request batch with system_prompt attribute.
        bot_task: Task type ("", "think", "recaption", "think_recaption").
        sys_type: System prompt selection mode:
            "None" - no system prompt
            "en_vanilla" - basic T2I prompt
            "en_recaption" - recaption prompt
            "en_think_recaption" - think+recaption prompt
            "en_unified" - unified multimodal prompt
            "dynamic" - auto-select based on bot_task
            "custom" - use user-provided system_prompt
    """
    custom_prompt = getattr(batch, "system_prompt", "") or ""

    if sys_type == "None":
        prompt = None
    elif sys_type == "en_unified":
        prompt = unified_system_prompt_en
    elif sys_type in t2i_system_prompts:
        prompt = t2i_system_prompts[sys_type]
    elif sys_type == "dynamic":
        if bot_task in ("think", "think_recaption"):
            prompt = t2i_system_prompts["en_think_recaption"]
        elif bot_task == "recaption":
            prompt = t2i_system_prompts["en_recaption"]
        elif bot_task == "image":
            prompt = t2i_system_prompt_en_vanilla
        else:
            prompt = custom_prompt
    elif sys_type == "custom":
        prompt = custom_prompt
    else:
        raise NotImplementedError(f"Unsupported system prompt type: {sys_type}")

    return prompt.strip() if prompt is not None else None


# ---------------------------------------------------------------------------
# HunyuanImage3BeforeDenoisingStage
# ---------------------------------------------------------------------------


class HunyuanImage3BeforeDenoisingStage(PipelineStage):
    """BeforeDenoisingStage for HunyuanImage-3.0 (Hybrid style).

    Consolidates AR text generation and all denoising preprocessing into a
    single stage, following the sglang-diffusion Hybrid pattern for models
    with complex pre-processing (AR token generation).

    When ``batch.bot_task`` is set (think / recaption / think_recaption), the
    stage first runs autoregressive text generation via the same transformer,
    then proceeds with the standard before-denoising pipeline.

    Prepares all inputs needed by DenoisingStage:
    - (Optional) AR CoT text + ratio prediction
    - Token sequences (conditional + unconditional) via HunyuanImage3Tokenizer
    - Token embeddings via transformer.embed_tokens
    - 2D RoPE (cos, sin) via build_batch_2d_rope
    - Initial noise latents
    - Timesteps via FlowMatchEulerDiscreteScheduler
    - All masks and scatter indices
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

    # parallelism_type: inherited REPLICATED from PipelineStage base class.
    # All ranks run AR + preprocessing independently.  TP all-reduce
    # produces identical logits on every rank; with greedy sampling
    # (default ar_top_k=1 → argmax) each rank independently picks the
    # same token, so no cross-rank broadcast is needed (ref: vllm-omni).

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

    def _cot_stop_token_ids(self, bot_task, need_ratio):
        tw = self.tokenizer_wrapper
        if need_ratio:
            return tw.get_ratio_token_ids()
        if bot_task in ("recaption", "think_recaption"):
            stop_ids = [tw.end_of_recaption_token_id]
        elif bot_task == "think":
            stop_ids = [tw.end_of_think_token_id, tw.end_of_recaption_token_id]
        else:
            stop_ids = [tw.eos_token_id]
        return [token_id for token_id in stop_ids if token_id is not None]

    def _build_text_generation_inputs(
        self,
        input_ids: torch.Tensor,
        dit_config,
        device: torch.device,
        position_ids: torch.Tensor | None = None,
        image_infos=None,
    ):
        """Build inputs for AR text generation.

        Returns (input_ids, rope_2d, position_ids).
        The causal mask is built internally by ``generate_text()``
        so it is NOT included here. ``image_infos`` (ti2i) gives the
        cond-image region a proper 2D spatial RoPE; ``None`` for t2i.
        """
        input_ids = input_ids.to(device)
        seq_len = input_ids.shape[1]
        n_elem = dit_config.rope_axes_dim[0] + dit_config.rope_axes_dim[1]
        rope_2d = build_batch_2d_rope(
            seq_len=seq_len,
            n_elem=n_elem,
            image_infos=image_infos,
            device=device,
            base=dit_config.rope_theta,
        )
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        else:
            position_ids = position_ids.to(device)
        return input_ids, rope_2d, position_ids

    def _build_ratio_logits_processor(self, base_size: int):
        tw = self.tokenizer_wrapper
        img_size_token_id = tw.special_token_map.get(f"<img_size_{base_size}>", None)
        ratio_token_ids = tw.get_ratio_token_ids()
        if img_size_token_id is None or not ratio_token_ids:
            return None
        return ConditionalSliceVocabLogitsProcessor(
            trigger_token_id=img_size_token_id,
            allowed_token_ids=ratio_token_ids,
            force_greedy=True,
        )

    def _build_ar_logits_processor(self, bot_task, template, base_size, need_ratio):
        processors = []
        transitions = self._build_stage_transitions(
            bot_task, template, base_size, need_ratio
        )
        if transitions:
            processors.append(StageTransitionLogitsProcessor(transitions))
        ratio_processor = self._build_ratio_logits_processor(base_size)
        if ratio_processor is not None:
            processors.append(ratio_processor)
        return _compose_logits_processors(processors)

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

    def _run_on_cfg_rank0(self, fn):
        """Run *fn* only on cfg_rank==0 and broadcast the result.

        In TP+CFG mode (cfg_world_size > 1), only cfg_rank==0 executes
        *fn*; the return value is broadcast to all other cfg_ranks so
        that every rank ends up with identical results.

        In non-CFG mode (single-GPU or pure-TP), *fn* runs once with
        no broadcast overhead.

        Args:
            fn: A zero-argument callable executed only on cfg_rank==0.

        Returns:
            Whatever *fn* returns (identical on all cfg_ranks after
            broadcast).
        """
        from sglang.multimodal_gen.runtime.distributed.parallel_state import (
            get_classifier_free_guidance_rank,
            get_classifier_free_guidance_world_size,
            get_cfg_group,
        )

        cfg_rank = get_classifier_free_guidance_rank()
        cfg_world_size = get_classifier_free_guidance_world_size()

        result = fn() if cfg_rank == 0 else None

        if cfg_world_size > 1:
            result = get_cfg_group().broadcast_object(result, src=0)

        return result

    # ------------------------------------------------------------------
    # AR generation (conditional sub-step)
    # ------------------------------------------------------------------

    def _run_ar_generation(self, batch: Req, server_args: ServerArgs):
        """Run AR text generation when bot_task is set.

        Generates CoT text (think / recaption) and predicts the image
        aspect ratio (ratio_index).  Results are written back to
        ``batch.cot_text`` and ``batch.ratio_index`` for use by the
        downstream token-sequence construction.

        Only cfg_rank==0 runs the actual AR generation; the result is
        broadcast to all other cfg_ranks via ``_run_on_cfg_rank0``.
        """
        bot_task = self._normalize_bot_task(getattr(batch, "bot_task", ""))
        if bot_task not in ("think", "recaption", "think_recaption"):
            return

        cot_text, ratio_index = self._run_on_cfg_rank0(
            lambda: self._do_ar_generation(batch, server_args, bot_task)
        )

        batch.cot_text = cot_text
        if ratio_index is not None:
            batch.ratio_index = ratio_index

        logger.info(
            "AR generation: bot_task=%s, cot_text_len=%d, ratio_index=%s",
            bot_task,
            len(cot_text),
            ratio_index,
        )

    def _do_ar_generation(self, batch: Req, server_args: ServerArgs, bot_task: str):
        """Execute AR generation on cfg_rank==0. Returns [cot_text, ratio_index]."""
        device = get_local_torch_device()
        dit_config = server_args.pipeline_config.dit_config.arch_config

        # 1. Read parameters
        template = getattr(batch, "sequence_template", "pretrain") or "pretrain"
        sys_type = getattr(batch, "sys_type", "dynamic") or "dynamic"
        base_size = dit_config.image_base_size
        need_ratio = self._is_auto_image_size(getattr(batch, "image_size", "auto"))

        # ti2i: feed the cond image into the AR context so it recaptions the
        # real subject. cond features were computed in forward() step -1.
        is_ti2i = getattr(batch, "image_path", None) is not None
        cond_image_info = (
            batch.cond_image_infos[0]
            if is_ti2i and getattr(batch, "cond_image_infos", None)
            else None
        )
        # ti2i must use the unified prompt (TI2I editing guidelines that
        # tell the model to describe the reference image's real subject and
        # preserve unchanged elements). The t2i think_recaption prompt
        # rewrites text only and would make the model hallucinate a subject.
        # Matches vllm-omni: it2i_* -> "en_unified".
        if is_ti2i and sys_type == "dynamic":
            sys_type = "en_unified"
        system_prompt = _resolve_system_prompt(batch, bot_task, sys_type)

        # 2. Build AR input sequence (ti2i inserts the cond-image region)
        prompt = _first_prompt(batch.prompt)
        ar_input = self.tokenizer_wrapper.build_ar_sequence(
            prompt=prompt,
            system_prompt=system_prompt,
            bot_task=bot_task,
            template=template,
            base_size=base_size,
            is_ti2i=is_ti2i,
            cond_image_info=cond_image_info,
        )

        # 3. Build generation inputs (ti2i gets 2D RoPE over the image region)
        input_ids, rope_2d, position_ids = self._build_text_generation_inputs(
            ar_input["input_ids"],
            dit_config,
            device,
            position_ids=ar_input["position_ids"],
            image_infos=ar_input.get("image_infos"),
        )
        logits_processor = self._build_ar_logits_processor(
            bot_task, template, base_size, need_ratio
        )

        # 3b. ti2i cond features + scatter indices for vision injection.
        # Scatter indices are unsqueezed to [1, N] to match the embeds' batch dim.
        ar_kwargs = {}
        if is_ti2i and cond_image_info is not None:
            ar_kwargs = dict(
                cond_vae_images=batch.cond_vae_images.to(device),
                cond_timestep=batch.cond_timestep.to(device),
                cond_vae_scatter_index=self._batch_to_device(
                    ar_input["cond_vae_scatter_index"], device
                ),
                cond_vit_embeds=batch.cond_vit_embeds.to(device),
                cond_vit_scatter_index=self._batch_to_device(
                    ar_input["cond_vit_scatter_index"], device
                ),
                cond_timestep_scatter_index=self._batch_to_device(
                    ar_input["cond_timestep_scatter_index"], device
                ),
            )

        # 4. Run AR generation
        max_new_tokens = getattr(batch, "ar_max_new_tokens", 2048)
        progress_log_interval = getattr(batch, "ar_progress_log_interval", 16)
        if (
            server_args.is_dit_layerwise_offload_selected
            and max_new_tokens > 128
        ):
            logger.warning(
                "HunyuanImage3 AR generation is using layerwise DiT offload "
                "with max_new_tokens=%d. This keeps memory low but can be slow "
                "because every generated token executes all DiT layers with "
                "CPU-to-device layer transfers.",
                max_new_tokens,
            )

        # 3c. ti2i: build a hybrid attention mask (causal for text,
        # bidirectional for the cond-image region). The AR was trained
        # with bidirectional image attention; a plain causal mask
        # prevents image patches from attending to each other across
        # raster order, destroying spatial coherence.
        prefill_attention_mask = None
        if is_ti2i:
            cond_image_slices = ar_input.get("cond_image_slices")
            if cond_image_slices:
                prefill_attention_mask = self._build_encoder_attention_mask(
                    seq_len=input_ids.shape[1],
                    image_slices=cond_image_slices,
                    device=device,
                    dtype=torch.bool,
                )

        generated_ids = self.transformer.generate_text(
            input_ids=input_ids,
            rope_2d=rope_2d,
            position_ids=position_ids,
            max_new_tokens=max_new_tokens,
            temperature=getattr(batch, "ar_temperature", 1.0),
            top_p=getattr(batch, "ar_top_p", 1.0),
            top_k=getattr(batch, "ar_top_k", 1),
            eos_token_id=self.tokenizer_wrapper.eos_token_id,
            stop_token_ids=self._cot_stop_token_ids(bot_task, need_ratio),
            logits_processor=logits_processor,
            progress_log_interval=progress_log_interval,
            image_infos=ar_input.get("image_infos"),
            prefill_attention_mask=prefill_attention_mask,
            **ar_kwargs,
        )

        # 5. Parse output: extract cot_text and ratio_index
        input_len = input_ids.shape[1]
        cot_text, ratio_index = self._parse_ar_output(
            generated_ids[0], input_len, bot_task
        )
        if _is_rank0():
            gen_tokens = generated_ids[0][input_len:]
            gen_token_ids = gen_tokens.tolist()
            gen_token_strs = self.tokenizer_wrapper.tokenizer.convert_ids_to_tokens(
                gen_token_ids
            )
            raw_decoded = self.tokenizer_wrapper.tokenizer.decode(
                gen_token_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            logger.info(
                "AR generation raw output: token_ids=%s token_strs=%s raw_decoded=%r parsed_cot=%r",
                gen_token_ids,
                gen_token_strs,
                raw_decoded,
                cot_text,
            )
        return [cot_text, ratio_index]

    def _run_ratio_prediction(self, batch: Req, server_args: ServerArgs):
        """Run a minimal AR generation to predict the image aspect ratio.

        This is called when bot_task is empty but image_size="auto",
        meaning we need to predict the ratio but don't need CoT text.

        Only cfg_rank==0 runs the actual prediction; the result is
        broadcast to all other cfg_ranks via ``_run_on_cfg_rank0``.
        """
        ratio_index = self._run_on_cfg_rank0(
            lambda: self._do_ratio_prediction(batch, server_args)
        )

        if ratio_index is not None:
            batch.ratio_index = ratio_index

        logger.info("Ratio prediction: ratio_index=%s", ratio_index)

    def _do_ratio_prediction(self, batch: Req, server_args: ServerArgs):
        """Execute ratio prediction on cfg_rank==0. Returns ratio_index or None."""
        device = get_local_torch_device()
        dit_config = server_args.pipeline_config.dit_config.arch_config
        tw = self.tokenizer_wrapper
        base_size = dit_config.image_base_size

        # 1. Read parameters
        template = getattr(batch, "sequence_template", "pretrain") or "pretrain"
        sys_type = getattr(batch, "sys_type", "dynamic") or "dynamic"
        system_prompt = _resolve_system_prompt(batch, "image", sys_type)

        # 2. Build minimal AR input with the official img_ratio prefix.
        prompt = _first_prompt(batch.prompt)
        ar_input = tw.build_ar_sequence(
            prompt=prompt,
            system_prompt=system_prompt or "",
            bot_task="",
            template=template,
            base_size=base_size,
        )
        img_size_token_id = tw.special_token_map.get(f"<img_size_{base_size}>", None)
        if img_size_token_id is None or tw.boi_token_id is None:
            raise ValueError(
                f"Missing image ratio prefix token(s) for base_size={base_size}"
            )

        ratio_prefix = []
        if template == "instruct" and tw.answer_token_id is not None:
            ratio_prefix.append(tw.answer_token_id)
        ratio_prefix.extend([tw.boi_token_id, img_size_token_id])

        input_ids = ar_input["input_ids"]
        prefix_tensor = torch.tensor([ratio_prefix], dtype=input_ids.dtype)
        input_ids = torch.cat([input_ids, prefix_tensor], dim=1)
        input_ids, rope_2d, position_ids = self._build_text_generation_inputs(
            input_ids,
            dit_config,
            device,
        )

        # 3. Restrict the next token after <img_size_*> to ratio tokens.
        ratio_token_ids = tw.get_ratio_token_ids()
        logits_processor = self._build_ratio_logits_processor(base_size)

        # 4. Run AR generation with small max_new_tokens
        generated_ids = self.transformer.generate_text(
            input_ids=input_ids,
            rope_2d=rope_2d,
            position_ids=position_ids,
            max_new_tokens=1,
            temperature=1.0,
            top_p=1.0,
            top_k=1,
            eos_token_id=tw.eos_token_id,
            stop_token_ids=ratio_token_ids if ratio_token_ids else None,
            logits_processor=logits_processor,
            progress_log_interval=getattr(batch, "ar_progress_log_interval", 16),
        )

        # 5. Extract ratio_index from generated tokens
        input_len = input_ids.shape[1]
        gen_tokens = generated_ids[0][input_len:]
        ratio_index = None
        for token_id in gen_tokens.tolist():
            if token_id in ratio_token_ids:
                ratio_index = tw.get_ratio_index_from_token_id(token_id)
                break

        return ratio_index

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
                elif hasattr(vae_result, "latent_dist"):
                    # diffusers EncoderOutput wrapper
                    _gen = torch.Generator(device=device).manual_seed(0)
                    vae_latent = vae_result.latent_dist.sample(_gen)
                else:
                    # AutoencoderKLConv3D returns the distribution directly
                    _gen = torch.Generator(device=device).manual_seed(0)
                    vae_latent = vae_result.sample(_gen)
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

                # Run vision encoder + aligner in fp32 for numerical stability.
                # The vision tower is loaded in bf16; on certain inputs (e.g. a
                # 1024x1024 image that fills the full 32x32 patch grid with no
                # padding) bf16 can produce NaN — typically attention-score
                # overflow or LayerNorm variance underflow — which then silently
                # poisons the AR recaption and the DiT denoising loop.  Casting
                # to fp32 for this single forward is cheap (the vision tower
                # runs once per generation) and avoids the NaN.
                vm_dtype = next(vision_model.parameters()).dtype
                va_dtype = (
                    next(vision_aligner.parameters()).dtype
                    if vision_aligner is not None
                    else None
                )
                vision_model.float()
                if vision_aligner is not None:
                    vision_aligner.float()
                with torch.autocast(device_type=device.type, enabled=False):
                    vision_output = vision_model(
                        pixel_values.float(),
                        attention_mask=pixel_attention_mask.unsqueeze(0),
                        spatial_shapes=spatial_shapes.unsqueeze(0),
                    )
                    image_embed = vision_output.last_hidden_state
                    if vision_aligner is not None:
                        image_embed = vision_aligner(image_embed)
                vision_model.to(vm_dtype)
                if vision_aligner is not None:
                    vision_aligner.to(va_dtype)

                if torch.isnan(image_embed).any():
                    raise RuntimeError(
                        "HunyuanImage3 vision encoder produced NaN even in fp32 "
                        f"(input image size {pil_image.size}). This is likely an "
                        "NPU kernel issue; try a different input aspect ratio."
                    )

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

    def _build_stage_transitions(self, bot_task, template, base_size, need_ratio=True):
        """Build stage transitions config based on bot_task and template."""
        transitions = {}
        tw = self.tokenizer_wrapper

        if bot_task in ("think", "think_recaption"):
            # After thinking, continue to recaption or to ratio prediction.
            if tw.end_of_think_token_id is not None:
                if bot_task == "think_recaption" and tw.recaption_token_id is not None:
                    transitions[tw.end_of_think_token_id] = [tw.recaption_token_id]
                elif need_ratio and tw.boi_token_id is not None:
                    img_size_id = tw.special_token_map.get(f"<img_size_{base_size}>")
                    forced = (
                        [tw.answer_token_id, tw.boi_token_id]
                        if template == "instruct" and tw.answer_token_id is not None
                        else [tw.boi_token_id]
                    )
                    if img_size_id is not None:
                        forced.append(img_size_id)
                    transitions[tw.end_of_think_token_id] = forced

        if need_ratio and bot_task in ("recaption", "think_recaption"):
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

    def _parse_ar_output(self, generated_ids, input_len, bot_task=None):
        """Parse AR output to extract cot_text and ratio_index.

        Uses input_len to identify where the input prefix ends, then extracts
        generated tokens. Trailing structural tokens (<answer>, <boi>,
        <img_size_*>, <img_ratio_*>) are stripped to isolate the CoT text.

        Args:
            generated_ids: 1D tensor of token IDs (input + generated).
            input_len: Length of the input prefix in generated_ids.

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

        # Extract generated portion (after the input prefix)
        gen_tokens = generated_ids[input_len:]
        # Remove trailing ratio token from decoded text
        if ratio_index is not None and len(gen_tokens) > 0:
            gen_tokens = gen_tokens[:-1]

        # Strip trailing structural tokens that are not part of CoT
        gen_tokens = self._strip_trailing_structural_tokens(gen_tokens)

        # Decode generated tokens to text
        cot_text = tw.tokenizer.decode(
            gen_tokens.tolist(),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if cot_text:
            if bot_task in ("think", "think_recaption") and not cot_text.startswith(
                tw.think_token
            ):
                cot_text = tw.think_token + cot_text
            elif bot_task == "recaption" and not cot_text.startswith(
                tw.recaption_token
            ):
                cot_text = tw.recaption_token + cot_text

        # Truncate at CoT end markers
        cot_text = self._truncate_at_cot_end(cot_text)

        return cot_text, ratio_index

    def _strip_trailing_structural_tokens(self, gen_tokens):
        """Strip trailing structural tokens from generated token sequence.

        Removes trailing <answer>, <boi>, <img_size_*>, <img_ratio_*>
        tokens that were forced by StageTransitionLogitsProcessor but are
        not part of the CoT content.
        """
        tw = self.tokenizer_wrapper
        # Build set of structural token IDs to strip from the tail
        structural_ids = set()
        if tw.answer_token_id is not None:
            structural_ids.add(tw.answer_token_id)
        if tw.boi_token_id is not None:
            structural_ids.add(tw.boi_token_id)
        # Add all <img_size_*> and <img_ratio_*> token IDs
        for key, tid in tw.special_token_map.items():
            if key.startswith("<img_size_") or key.startswith("<img_ratio_"):
                structural_ids.add(tid)

        # Strip from the tail
        end = len(gen_tokens)
        while end > 0 and gen_tokens[end - 1].item() in structural_ids:
            end -= 1
        return gen_tokens[:end]

    @staticmethod
    def _truncate_at_cot_end(cot_text):
        """Truncate AR output at first CoT end marker.

        Mirrors vllm-omni's _truncate_at_cot_end: the trailing
        <answer><boi><img_size_*><img_ratio_*> is consumed via
        height/width extraction and must not leak into the DiT prompt.
        """
        for marker in ("</recaption>", "</think>", "ebil_think"):
            idx = cot_text.find(marker)
            if idx != -1:
                return cot_text[: idx + len(marker)]
        return cot_text

    @staticmethod
    def _apply_drop_think(cot_text, drop_think):
        """Strip the think portion from CoT text when drop_think=True.

        Handles formats like:
            <think>think_content</think><recaption>recaption_content</recaption>
        When drop_think=True, strips the <think_token>...ebil_think portion and
        keeps only <recaption>...</recaption>.
        """
        if not drop_think or not cot_text:
            return cot_text

        for think_end_marker in ("</think>", "ebil_think"):
            think_end = cot_text.find(think_end_marker)
            if think_end != -1:
                return cot_text[think_end + len(think_end_marker):]

        return cot_text

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
        image_size = getattr(batch, "image_size", "auto") or "auto"
        requested_width, requested_height, requested_ratio_index = (
            self._parse_requested_image_size(image_size)
        )
        if requested_ratio_index is not None:
            batch.ratio_index = requested_ratio_index

        # -1. Process condition images (ti2i) BEFORE AR generation. The AR
        # recaption must see the input image so it describes the real subject
        # instead of hallucinating one. _process_cond_images depends only on
        # the input image (it buckets by the image's own size), not on the
        # output gen size or ratio_index, so it is safe to run here. The
        # results are stashed on the batch and reused at step 5 below.
        is_ti2i = getattr(batch, "image_path", None) is not None
        if is_ti2i:
            (
                batch.cond_vae_images,
                batch.cond_timestep,
                batch.cond_vit_embeds,
                batch.cond_image_infos,
            ) = self._process_cond_images(batch, device)
        else:
            batch.cond_vae_images = None
            batch.cond_timestep = None
            batch.cond_vit_embeds = None
            batch.cond_image_infos = None

        # 0. Optional AR generation (Hybrid: consolidated into this stage)
        self._run_ar_generation(batch, server_args)

        # [EXPERIMENT] Force a pure *description* recaption (instead of the
        # AR-generated edit-instruction) to test whether the recaption format
        # is why the text edit fails to apply.  Gated by env var; remove after.
        import os as _os_desc
        if is_ti2i and _os_desc.environ.get("HY3_TI2I_DESC_RECPTION"):
            batch.cot_text = (
                "<recaption>一只橘色的家猫端坐在厚厚的积雪中。猫咪的毛发呈现橘色与白色相间的条纹花纹，"
                "浓密蓬松，双眼明亮有神，鼻子粉红。背景是广阔的冬日雪景，地面被洁白的积雪完全覆盖，"
                "远处有被雪点缀的松树和灌木丛。低角度的金色阳光从右侧斜照过来，"
                "在雪地和猫咪身上形成温暖柔和的金色光影，与雪的冷白色调形成对比，营造温馨的冬日氛围。"
                "空中飘洒着细小的雪花，部分散落在猫咪的毛发上。</recaption>"
            )
            logger.info("[EXPERIMENT] forced description recaption for ti2i")

        # 0b. If no ratio_index predicted yet and image_size="auto", run ratio prediction
        if self._is_auto_image_size(image_size) and (
            not hasattr(batch, "ratio_index") or batch.ratio_index is None
        ):
            self._run_ratio_prediction(batch, server_args)

        # 1. Parse inputs from batch
        prompt = _first_prompt(batch.prompt)
        cot_text = getattr(batch, "cot_text", "") or ""
        bot_task = getattr(batch, "bot_task", "") or ""
        sys_type = getattr(batch, "sys_type", "dynamic") or "dynamic"
        system_prompt = _resolve_system_prompt(batch, bot_task, sys_type) or ""
        template = getattr(batch, "sequence_template", "pretrain") or "pretrain"
        drop_think = getattr(batch, "drop_think", False)

        # 1a. Apply drop_think: strip think portion from cot_text
        original_cot_text = cot_text
        cot_text = self._apply_drop_think(cot_text, drop_think)
        if (
            drop_think
            and system_prompt
            and self.tokenizer_wrapper.think_token in original_cot_text
            and self.tokenizer_wrapper.recaption_token in cot_text
        ):
            system_prompt = _resolve_system_prompt(batch, bot_task, "en_recaption") or ""

        height = batch.height
        width = batch.width
        if requested_width is not None and requested_height is not None:
            width = requested_width
            height = requested_height
        guidance_scale = batch.guidance_scale
        num_inference_steps = batch.num_inference_steps
        seed = batch.seed

        # 1b. If AR generation predicted ratio_index, override height/width
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

        # 5. Reuse condition features computed before AR generation (step -1).
        cond_vae_images = batch.cond_vae_images
        cond_timestep = batch.cond_timestep
        cond_vit_embeds = batch.cond_vit_embeds
        cond_image_infos = batch.cond_image_infos

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

        # [HY3-DBG] verify cond vs uncond token sequences actually differ
        import os as _os
        if _os.environ.get("HY3_DEBUG_TI2I") and do_cfg:
            _tok = self.tokenizer_wrapper.tokenizer
            _cdec = _tok.convert_ids_to_tokens(cond_tokens[0][:40].tolist())
            _udec = _tok.convert_ids_to_tokens(uncond_tokens[0][:40].tolist())
            _ce = cond_embeds.float()
            _ue = uncond_embeds.float()
            _ediff = (_ce - _ue).norm().item()
            print(
                f"[HY3-DBG-SEQ] cond_seq_len={cond_tokens.shape[1]} "
                f"uncond_seq_len={uncond_tokens.shape[1]} "
                f"token_diff={int((cond_tokens[0] != uncond_tokens[0]).sum().item())} "
                f"embed_diff_norm={_ediff:.2f}/{_ce.norm().item():.2f} "
                f"cond_first40={_cdec[:18]} "
                f"uncond_first40={_udec[:18]}",
                flush=True,
            )

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
        # Build encoder attention mask using joint_full strategy (vllm-omni):
        # - ti2i: joint (vae+sep+vit combined) + gen as bidirectional regions
        # - t2i: gen only (no joint region)
        if cond_output.joint_image_slices:
            cond_attn_slices = cond_output.joint_image_slices + cond_output.gen_image_slices
        else:
            cond_attn_slices = cond_output.gen_image_slices or cond_output.all_image_slices
        encoder_attention_mask = self._build_encoder_attention_mask(
            seq_len, cond_attn_slices, device, dtype
        )
        if do_cfg:
            # Uncond ti2i keeps same joint structure (placeholders) with null features
            if uncond_output.joint_image_slices:
                uncond_attn_slices = uncond_output.joint_image_slices + uncond_output.gen_image_slices
            else:
                uncond_attn_slices = uncond_output.gen_image_slices or uncond_output.all_image_slices
            neg_encoder_attention_mask = self._build_encoder_attention_mask(
                neg_seq_len, uncond_attn_slices, device, dtype
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
        if do_cfg:
            neg_cond_timestep_scatter_index = self._batch_to_device(
                uncond_output.cond_timestep_scatter_index, device
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
        # ti2i CFG: the source image is a FIXED condition present in BOTH
        # cond and uncond branches (only the text differs). vllm-omni repeats
        # cond images cfg_factor times for the same reason. So we need the
        # uncond branch's cond masks/scatter indices too (positions differ
        # because the uncond text length differs).
        if do_cfg:
            neg_cond_vae_image_mask = self._batch_to_device(
                uncond_output.cond_vae_image_mask, device
            )
            neg_cond_vae_scatter_index = self._batch_to_device(
                uncond_output.cond_vae_scatter_index, device
            )
            neg_cond_vit_image_mask = self._batch_to_device(
                uncond_output.cond_vit_image_mask, device
            )
            neg_cond_vit_scatter_index = self._batch_to_device(
                uncond_output.cond_vit_scatter_index, device
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
        if do_cfg:
            batch.neg_cond_timestep_scatter_index = neg_cond_timestep_scatter_index
            batch.neg_cond_vae_image_mask = neg_cond_vae_image_mask
            batch.neg_cond_vae_scatter_index = neg_cond_vae_scatter_index
            batch.neg_cond_vit_image_mask = neg_cond_vit_image_mask
            batch.neg_cond_vit_scatter_index = neg_cond_vit_scatter_index
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
