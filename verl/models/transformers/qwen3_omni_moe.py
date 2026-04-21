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
from typing import Iterable

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_PATCHED_SENTINEL = "_verl_all_experts_patched"


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
