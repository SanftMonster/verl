# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Monkey patches for the Qwen3-Omni MoE model family.

Upstream ``Qwen3OmniMoeThinkerTextSparseMoeBlock.forward`` (and the Talker
variant) iterates only over experts that actually received tokens on the
current rank (``for expert_idx in expert_hit``). Under FSDP2 this produces
a *different* autograd graph on each rank whenever the set of activated
experts differs between ranks. FSDP2 only queues a ReduceScatter for
parameters that received gradients, so ranks end up launching
ReduceScatters with incompatible inputs. The gradients get silently
corrupted and the symptom surfaces later as

    torch.utils.checkpoint.CheckpointError: A different number of tensors
    was saved during the original forward and recomputation.

See verl-project/verl#3258 and pytorch/pytorch#171355 for the full
discussion.

The patch below preserves the numeric behaviour of the original forward
(the contribution of inactive experts is still zero) while ensuring every
expert always participates in autograd. That keeps the gradient
bookkeeping identical across ranks so FSDP2 can safely issue
ReduceScatter and activation checkpointing can re-run the forward
deterministically.
"""

from __future__ import annotations

import functools
import logging
from typing import Iterable, Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_PATCHED_SENTINEL = "_verl_all_experts_patched"


def get_rope_index(
    processor,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    image_grid_thw: Optional[torch.Tensor] = None,
    video_grid_thw: Optional[torch.Tensor] = None,
    use_audio_in_video: bool = False,
    audio_seqlens: Optional[torch.Tensor] = None,
    second_per_grids: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Audio-aware MRope for Qwen3-Omni.

    HF's ``Qwen3OmniMoeThinkerForConditionalGeneration.get_rope_index`` gates
    the 3D audio/image/video-aware branch on ``image_grid_thw is not None or
    video_grid_thw is not None``. Audio-only samples fall through to a 1D
    ``cumsum`` broadcast across 3 axes, which corrupts MRope at the audio
    region and makes actor logprobs diverge from the rollout engine.

    This helper reconstructs the audio-aware branch locally: when
    ``audio_seqlens`` is present it walks the pre-expanded ``input_ids`` and
    assigns per-axis MRope positions around ``audio_start / audio_pad*N /
    audio_end`` blocks. When images/videos are present, it delegates back to
    the processor's bound ``get_rope_index`` (which works correctly once at
    least one vision grid is supplied, provided ``get_llm_pos_ids_for_vision``
    is also bound by ``verl.utils.tokenizer.hf_processor``).

    Returns ``(position_ids, mrope_position_deltas)`` matching HF's
    signature: ``position_ids`` has shape ``(3, batch, seq_len)`` and
    ``dtype=torch.long``.
    """
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
        _get_feat_extract_output_lengths,
    )

    has_vision = image_grid_thw is not None or video_grid_thw is not None
    has_audio = audio_seqlens is not None

    # Vision branch: HF's bound implementation already works. Fall back to
    # it when any vision grid is provided — the guard fires correctly.
    if has_vision:
        return processor.get_rope_index(
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            use_audio_in_video=use_audio_in_video,
            audio_seqlens=audio_seqlens,
            second_per_grids=second_per_grids,
        )

    # Pure-text branch: HF fallback is correct (the 1D linear cumsum is what
    # a text-only sample needs). Cast to long for consistency.
    if not has_audio:
        pos, deltas = processor.get_rope_index(
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_grid_thw=None,
            video_grid_thw=None,
            use_audio_in_video=use_audio_in_video,
            audio_seqlens=None,
            second_per_grids=second_per_grids,
        )
        return pos.long(), deltas

    # Audio-only branch: replicate HF's audio-aware loop but without the
    # buggy guard. Mirrors transformers 4.57.1
    # ``Qwen3OmniMoeThinkerForConditionalGeneration.get_rope_index``.
    cfg = getattr(processor, "config", None)
    audio_token_id = getattr(processor, "audio_token_id", None)
    audio_start_token_id = getattr(processor, "audio_start_token_id", None)
    audio_end_token_id = getattr(processor, "audio_end_token_id", None)
    if cfg is not None:
        audio_token_id = getattr(cfg, "audio_token_id", audio_token_id)
        audio_start_token_id = getattr(cfg, "audio_start_token_id", audio_start_token_id)
        audio_end_token_id = getattr(cfg, "audio_end_token_id", audio_end_token_id)
    if audio_token_id is None or audio_start_token_id is None or audio_end_token_id is None:
        raise RuntimeError(
            "Qwen3-Omni audio MRope requires audio_token_id / audio_start_token_id / "
            "audio_end_token_id on the processor; did you forget ensure_qwen3_omni_processor_attrs?"
        )

    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    batch_size, seq_len = input_ids.shape
    position_ids = torch.zeros(3, batch_size, seq_len, dtype=torch.long, device=input_ids.device)
    mrope_position_deltas = []

    attn_bool = attention_mask == 1

    audio_idx_global = 0
    for i in range(batch_size):
        row_ids = input_ids[i][attn_bool[i]]
        row_tokens = row_ids.tolist()
        row_len = len(row_tokens)

        llm_pos_ids_list: list[torch.Tensor] = []
        st = 0
        remain_audios = int((row_ids == audio_start_token_id).sum())
        bos_len, eos_len = 1, 1

        for _ in range(remain_audios):
            st_idx = int(llm_pos_ids_list[-1].max()) + 1 if llm_pos_ids_list else 0
            try:
                ed_audio_start = row_tokens.index(audio_start_token_id, st)
            except ValueError:
                break

            # Text run before this audio block.
            text_len = ed_audio_start - st
            if text_len > 0:
                llm_pos_ids_list.append(torch.arange(text_len, dtype=torch.long).view(1, -1).expand(3, -1) + st_idx)
                st_idx += text_len

            # audio_start (bos_len=1).
            llm_pos_ids_list.append(torch.arange(bos_len, dtype=torch.long).view(1, -1).expand(3, -1) + st_idx)
            st_idx += bos_len

            # audio_pad*N_audio (N_audio from _get_feat_extract_output_lengths).
            audio_len_t = _get_feat_extract_output_lengths(audio_seqlens[audio_idx_global])
            audio_len = int(audio_len_t.item() if torch.is_tensor(audio_len_t) else audio_len_t)
            llm_pos_ids_list.append(torch.arange(audio_len, dtype=torch.long).view(1, -1).expand(3, -1) + st_idx)

            # advance st past text + audio_start + audio_pad*N + audio_end.
            st += int(text_len + bos_len + audio_len + eos_len)
            audio_idx_global += 1

            # audio_end (eos_len=1).
            st_idx = int(llm_pos_ids_list[-1].max()) + 1
            llm_pos_ids_list.append(torch.arange(eos_len, dtype=torch.long).view(1, -1).expand(3, -1) + st_idx)

        # Trailing text run.
        if st < row_len:
            st_idx = int(llm_pos_ids_list[-1].max()) + 1 if llm_pos_ids_list else 0
            tail_len = row_len - st
            llm_pos_ids_list.append(torch.arange(tail_len, dtype=torch.long).view(1, -1).expand(3, -1) + st_idx)
        elif not llm_pos_ids_list:
            # Attention-masked-to-zero or empty row.
            llm_pos_ids_list.append(torch.zeros(3, row_len, dtype=torch.long))

        llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
        position_ids[..., i, attn_bool[i]] = llm_positions.to(position_ids.device)
        mrope_position_deltas.append(int(llm_positions.max()) + 1 - row_len)

    mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
    return position_ids, mrope_position_deltas


def _all_experts_forward(self, hidden_states: torch.Tensor):
    """Drop-in replacement for ``Qwen3OmniMoe*TextSparseMoeBlock.forward``.

    The only behavioural difference versus the upstream forward is that we
    iterate over *every* expert regardless of whether any tokens were routed
    to it. For experts that received no tokens this amounts to running them
    on an empty batch, which is a no-op for the produced output but keeps
    their parameters in the autograd graph (with zero gradients) so FSDP2
    sees a consistent gradient set across ranks.
    """
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)

    router_logits = self.gate(hidden_states)
    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
    if getattr(self, "norm_topk_prob", True):
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(hidden_states.dtype)

    final_hidden_states = torch.zeros(
        (batch_size * sequence_length, hidden_dim),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

    # Iterate over *all* experts to ensure every rank registers gradients
    # for the full expert set. ``index_add_`` with an empty index tensor is
    # a no-op, so the arithmetic is unchanged.
    for expert_idx in range(self.num_experts):
        expert_layer = self.experts[expert_idx]
        idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))

        current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
        current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
        final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

    final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
    return final_hidden_states, router_logits


def _iter_sparse_moe_classes() -> Iterable[type]:
    """Return the Qwen3-Omni sparse MoE block classes that need patching.

    Classes are imported lazily so verl keeps working with transformers
    versions that do not yet ship Qwen3-Omni.
    """
    try:
        from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe as module
    except ImportError:
        return ()

    names = (
        "Qwen3OmniMoeThinkerTextSparseMoeBlock",
        # The Talker variant shares the same routing pattern; patch it as
        # well so Talker fine-tuning does not regress to the same bug.
        "Qwen3OmniMoeTalkerTextSparseMoeBlock",
    )
    classes = []
    for name in names:
        cls = getattr(module, name, None)
        if cls is not None:
            classes.append(cls)
    return tuple(classes)


def patch_qwen3_omni_moe_sparse_moe_block_forward() -> None:
    """Route every expert on every rank to keep FSDP2 gradient sets aligned.

    Must be called before the model is wrapped by FSDP. The patch is
    idempotent and cheap, so calling it multiple times is safe.
    """
    patched_any = False
    for cls in _iter_sparse_moe_classes():
        current_forward = cls.forward
        if getattr(current_forward, _PATCHED_SENTINEL, False):
            continue

        wrapped = functools.wraps(current_forward)(_all_experts_forward)
        setattr(wrapped, _PATCHED_SENTINEL, True)
        cls.forward = wrapped
        patched_any = True
        logger.info(
            "Monkey patched %s.forward to always route through every expert "
            "(required for FSDP2 + gradient checkpointing consistency).",
            cls.__name__,
        )

    if not patched_any:
        logger.debug("Qwen3-Omni MoE sparse block patch skipped (no matching classes found).")
