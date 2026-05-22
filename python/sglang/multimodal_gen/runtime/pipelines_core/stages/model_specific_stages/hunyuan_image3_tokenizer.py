"""HunyuanImage-3.0 tokenizer, sequence builder, and system prompts.

This module contains:

- System prompt constants and ``resolve_system_prompt``
- ``TokenizerOutput`` (NamedTuple)
- ``HunyuanImage3Tokenizer`` — token-sequence builder for the unified
  text + image + special-token sequences that the single-stream
  transformer processes.

Logits processors live in ``hunyuan_image3.py``.
"""

import random
from collections import defaultdict
from copy import deepcopy
from typing import NamedTuple

import torch

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Tokenizer output (NamedTuple — lightweight, immutable, no dataclass overhead)
# ---------------------------------------------------------------------------


class TokenizerOutput(NamedTuple):
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
        self.think_token = "<think_start>"
        self.end_of_think_token = "<think_end>"
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

        The source/cond image is placed in the **user turn** (right after the
        user prompt, before the Assistant prefix and recaption), matching the
        official message format
            user(text) -> user(joint_image) -> assistant(recaption) -> assistant(gen_image)
        and vllm-omni's it2i layout. This MUST stay consistent with
        ``build_ar_sequence``, which places the cond image at the same position
        so the AR recaption and the denoising forward see the source image at
        the same sequence location.

        Final instruct structure:
          <bos> [system] [User: <prompt> <boi>[cond_meta] <img>*vae_len
          <joint_img_sep> <img>*vit_len <eoi>] [Assistant: <recaption> <answer>
          <boi>[gen_meta] <img>*N <eoi> </answer>]

        Args:
            cond_image_infos: list of dicts, each with:
                vae_token_h, vae_token_w, vit_token_h, vit_token_w,
                base_size, ratio_idx
        """
        uncond_kwargs = dict(uncond_p=uncond_p)
        sections = []

        # User turn: system + "User: " prefix + prompt, then the cond image
        # BEFORE the "\n\nAssistant: " suffix (so the source image lives in the
        # user turn, ahead of the recaption). Inline-expands
        # ``_append_prompt_sections`` to inject cond images at the right spot.
        if template == "instruct":
            if system_prompt:
                sections.append(dict(type="text", text=system_prompt))
                sections.append(dict(type="text", text=self.instruct_user_sep))
            sections.append(dict(type="text", text=self.instruct_user_prefix))
            sections.append(dict(type="text", text=prompt, **uncond_kwargs))
        else:
            if system_prompt:
                sections.append(dict(type="text", text=system_prompt))
            sections.append(dict(type="text", text=prompt, **uncond_kwargs))

        # Add joint_image sections for each condition image, in the user turn
        # (after the prompt, before the Assistant prefix / recaption).
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

        # Close the user turn and open the assistant turn.
        if template == "instruct":
            sections.append(dict(type="text", text=self.instruct_user_sep))
            sections.append(dict(type="text", text=self.instruct_assistant_prefix))

        self._append_cot_sections(sections, cot_text, uncond_kwargs)
        if template == "instruct" and self.answer_token_id is not None:
            sections.append(dict(type="text", text="<answer>", ignore=True))

        # Generated image (assistant turn, after the recaption/<answer>).
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
        """Parse image slices and mask from extra_token_pos.

        Keys must match ``_build_token_sequence``'s convention
        ``<{prefix}>_start`` / ``<{prefix}>_end`` (chevron before
        ``_start``/``_end``) for all prefixes.
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
        ti2i it also returns the cond_* scatter indices and image slices so
        the caller can (a) inject vision features into the AR forward and
        (b) build 2D RoPE over the image region. The ti2i-only keys are
        ``None`` for t2i.
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
            "image_infos": image_infos,
        }

    def parse_cot_text(self, cot_text):
        """Parse CoT text into sections list for _encode_sections.

        Input: "<think_start>think content<think_end><recaption>recaption content</recaption>"
        Output: [
            dict(type="text", text="<think_start>think content<think_end>"),
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
# System prompts
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

1. Thinking Phase (<think_start>): In the <think_start> tag, you need to conduct a structured thinking process, progressively breaking down and enriching the constituent elements of the image. This process must include, but is not limited to, the following dimensions:

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
<think_start>Thinking process<think_end><recaption>Refined image description</recaption>Generate Image


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

#### **<think_start> Mode (Think + Rewrite)**:
*   **Trigger:** Input begins with `<think_start>`.
*   **Task:** First, conduct a structured analysis of the request within `<think_start>` tags. Then, output the professional prompt, rewritten based on the analysis, within `<recaption>` tags.
*   **Output:** Strictly adhere to the format: `<think_start>Analysis process<think_end><recaption>Rewritten prompt</recaption>`

---
### Execution Standards and Guidelines
#### **`<think_start>` Phase: Analysis Guidelines**
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


# ---------------------------------------------------------------------------
# System prompt resolver
# ---------------------------------------------------------------------------


def resolve_system_prompt(batch, bot_task="", sys_type="dynamic"):
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
