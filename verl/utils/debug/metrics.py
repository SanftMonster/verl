# Copyright 2025 Individual Contributor: TomQunChaoA
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

import logging
import os
from pathlib import Path

import torch

from verl.protocol import DataProto

logger = logging.getLogger(__file__)


# Module-level counter so consecutive invocations of ``calculate_debug_metrics``
# in the same training step produce distinct dump filenames. Gated entirely by
# the ``VERL_DUMP_DEBUG_METRICS_DIR`` env var, so the default training path
# pays zero cost.
_DUMP_STATE = {"counter": 0}


def calculate_token_list_diff(tensor1: torch.Tensor, tensor2: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # verify inputs
    if tensor1.numel() == 0 or tensor2.numel() == 0:
        return torch.zeros(tensor1.shape[0], dtype=torch.long, device=tensor1.device)
    if tensor1.shape != tensor2.shape or mask.shape != tensor1.shape or mask.shape != tensor2.shape:
        print(
            f"<WARN> dim of tensor1, tensor2, mask is not equal, {(tensor1.shape)=},{(tensor2.shape)=}, {(mask.shape)=}"
        )
        return torch.ones_like(tensor1)
    # transfer to same device
    if tensor2.device != tensor1.device:
        tensor2 = tensor2.to(tensor1.device)
    if mask.device != tensor1.device:
        mask = mask.to(tensor1.device)

    # calculate diff
    diff_mask = tensor1 != tensor2

    valid_diff_mask = diff_mask & (mask == 1)

    diff_counts = valid_diff_mask.sum(dim=1)

    return diff_counts


def pearson_correlation_coefficient(tensor1: torch.Tensor, tensor2: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # implemention of https://arxiv.org/pdf/2506.13585
    if tensor1.shape != tensor2.shape or mask.shape != tensor1.shape or mask.shape != tensor2.shape:
        return 0
    mt1 = torch.masked_select(tensor1, mask)
    mt2 = torch.masked_select(tensor2, mask)
    result = torch.corrcoef(torch.stack([mt1, mt2], dim=0))
    return result[0][1].detach().item()


def calculate_log_prob_diff(log_probs1: torch.Tensor, log_probs2: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    full_diff = torch.abs(log_probs1 - log_probs2)
    return torch.masked_select(full_diff, mask)


def _maybe_dump_debug_metrics(
    data: DataProto,
    rollout_old_log_probs: torch.Tensor,
    actor_old_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
) -> None:
    """Persist per-batch tensors to disk when ``VERL_DUMP_DEBUG_METRICS_DIR``
    is set.

    This is intended for targeted debugging of the
    ``training/rollout_probs_diff_*`` series — in particular, to let an
    offline clean-room actor/vLLM probe re-score the **same** ``input_ids``
    + ``multi_modal_inputs`` the live FSDP2/vLLM stack just saw. Disabled
    by default (env-gated) so zero cost is paid in normal training.

    Controls (env vars):
      - ``VERL_DUMP_DEBUG_METRICS_DIR`` — output directory (must be a
        shared path if worker & driver run on different nodes).
      - ``VERL_DUMP_DEBUG_METRICS_MAX_CALLS`` — stop after this many calls
        (default: 4). Caps disk usage during long runs that accidentally
        leave the env var set.
      - ``VERL_DUMP_DEBUG_METRICS_STEP`` — optional step tag to embed in
        filenames when ``data.meta_info["global_step"]`` is unavailable.
    """
    dump_dir = os.environ.get("VERL_DUMP_DEBUG_METRICS_DIR")
    if not dump_dir:
        return

    try:
        max_calls = int(os.environ.get("VERL_DUMP_DEBUG_METRICS_MAX_CALLS", "4"))
    except ValueError:
        max_calls = 4
    if _DUMP_STATE["counter"] >= max_calls:
        return

    try:
        Path(dump_dir).mkdir(parents=True, exist_ok=True)

        meta = getattr(data, "meta_info", None) or {}
        step_tag = meta.get("global_step") or meta.get("global_steps")
        if step_tag is None:
            step_tag = os.environ.get("VERL_DUMP_DEBUG_METRICS_STEP", "x")

        call_idx = _DUMP_STATE["counter"]
        _DUMP_STATE["counter"] += 1

        payload: dict = {
            "rollout_log_probs": rollout_old_log_probs.detach().cpu().contiguous(),
            "actor_old_log_probs": actor_old_log_probs.detach().cpu().contiguous(),
            "response_mask": response_mask.detach().cpu().contiguous(),
            "responses": data.batch["responses"].detach().cpu().contiguous(),
            "meta_info": dict(meta),
        }
        for key in ("input_ids", "attention_mask", "position_ids", "prompts"):
            if key in data.batch.keys():
                payload[key] = data.batch[key].detach().cpu().contiguous()

        nt = getattr(data, "non_tensor_batch", None) or {}
        if "multi_modal_inputs" in nt:
            mmi_raw = nt["multi_modal_inputs"]
            mmi_out = []
            for entry in mmi_raw:
                if isinstance(entry, dict):
                    cleaned: dict = {}
                    for k, v in entry.items():
                        if hasattr(v, "detach"):
                            cleaned[k] = v.detach().cpu().contiguous()
                        else:
                            cleaned[k] = v
                    mmi_out.append(cleaned)
                else:
                    mmi_out.append(entry)
            payload["multi_modal_inputs"] = mmi_out
        if "uid" in nt:
            try:
                payload["uid"] = list(nt["uid"])
            except Exception:
                pass
        # raw_prompt + index are the dataset-side handles that map each row
        # back to its source parquet record (see ``RLHFDataset.__getitem__``).
        # Offline probes need these to re-open the original audio/image files
        # for the vLLM clean-room forward (vLLM's ``multi_modal_data`` expects
        # raw audio arrays, not the pre-processed mel features we've already
        # captured in ``multi_modal_inputs``).
        for handle in ("raw_prompt", "index", "data_source"):
            if handle in nt:
                try:
                    payload[handle] = list(nt[handle])
                except Exception:
                    pass

        file_path = Path(dump_dir) / f"step_{step_tag}_call_{call_idx:03d}.pt"
        torch.save(payload, file_path)
        logger.warning(
            "[VERL_DUMP_DEBUG_METRICS_DIR] wrote %s (rollout=%s, actor=%s, batch_keys=%s, nt_keys=%s)",
            file_path,
            tuple(rollout_old_log_probs.shape),
            tuple(actor_old_log_probs.shape),
            list(data.batch.keys())[:8],
            list(nt.keys())[:8],
        )
    except Exception as exc:  # keep training running even if the dump fails
        logger.warning("[VERL_DUMP_DEBUG_METRICS_DIR] dump failed: %r", exc)


def calculate_debug_metrics(data: DataProto) -> dict:
    """
    calculate rollout vs actor logprobs diff, for debugging purpose

    Args:
        data: DataProto
            the data batch to calculate
            rollout_log_probs: log_probs record when rollout forward tokens
            old_log_probs(actor log probs): log_probs record when actor forward tokens
            loss_mask or attention_mask: to mask unrelated token
            responses: the response tokens, for calculating size
    Returns:
        dict: metrics
            "training/rollout_probs_diff_valid": 1->input is valid, 0->input is invalid
            "training/rollout_probs_diff_max": max value of logprob diff of rollout vs. actor
            "training/rollout_probs_diff_mean": mean value of logprob diff of rollout vs. actor
            "training/rollout_probs_diff_std": std value of logprob diff of rollout vs. actor
            "training/rollout_actor_probs_pearson_corr": logprob's pearson corrcoef of rollout vs. actor, reference to https://arxiv.org/pdf/2506.13585
    """

    rollout_old_log_probs = data.batch["rollout_log_probs"]
    actor_old_log_probs = data.batch["old_log_probs"]
    if "response_mask" in data.batch:
        logger.debug("response mask found, use it to mask log probs")
        log_prob_mask = data.batch["response_mask"]
    elif "attention_mask" in data.batch:
        log_prob_mask = data.batch["attention_mask"]
    else:
        logger.warning(f"no mask info found, use all log probs, {(data.batch.keys())=}")
        log_prob_mask = torch.ones_like(rollout_old_log_probs)
    responses = data.batch["responses"]
    response_length = responses.size(1)

    response_mask = log_prob_mask[:, -response_length:]
    # calculate pearson corrcoef
    actor_probs = torch.exp(actor_old_log_probs)
    rollout_probs = torch.exp(rollout_old_log_probs)
    response_mask_bool = response_mask.bool()

    # check if there are any valid tokens before computing metrics
    if not response_mask_bool.any():
        logger.warning("response_mask is all False, returning default metrics")
        return {
            "training/rollout_probs_diff_valid": 0,
            "training/rollout_probs_diff_max": float("nan"),
            "training/rollout_probs_diff_mean": float("nan"),
            "training/rollout_probs_diff_std": float("nan"),
            "training/rollout_actor_probs_pearson_corr": float("nan"),
        }

    pearson_corrcoef = pearson_correlation_coefficient(actor_probs, rollout_probs, response_mask_bool)
    rollout_probs_diff = calculate_log_prob_diff(actor_probs, rollout_probs, response_mask_bool)

    # Env-gated: persist the exact tensors we just compared so an offline
    # single-process probe can re-score the same inputs without FSDP2 /
    # multi-node NCCL noise. See ``_maybe_dump_debug_metrics`` for the
    # controlling env vars. No-op when ``VERL_DUMP_DEBUG_METRICS_DIR`` is
    # unset, so normal training is unaffected.
    _maybe_dump_debug_metrics(data, rollout_old_log_probs, actor_old_log_probs, response_mask)

    return {
        "training/rollout_probs_diff_valid": 1,
        "training/rollout_probs_diff_max": torch.max(rollout_probs_diff).detach().item(),
        "training/rollout_probs_diff_mean": torch.mean(rollout_probs_diff).detach().item(),
        "training/rollout_probs_diff_std": torch.std(rollout_probs_diff).detach().item(),
        "training/rollout_actor_probs_pearson_corr": pearson_corrcoef,
    }
