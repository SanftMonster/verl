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
import asyncio
import logging

import numpy as np
import uvicorn
from fastapi import FastAPI

from verl.utils.tokenizer import get_processor_token_id, is_qwen3_omni_processor

logger = logging.getLogger(__file__)


def get_max_position_embeddings(hf_config) -> int:
    max_len = getattr(hf_config, "max_position_embeddings", None)
    if max_len is None:
        text_config = getattr(hf_config, "text_config", None)
        if text_config is not None:
            max_len = getattr(text_config, "max_position_embeddings", None)
    if max_len is None:
        thinker_config = getattr(hf_config, "thinker_config", None)
        if thinker_config is not None:
            max_len = getattr(thinker_config, "max_position_embeddings", None)
            if max_len is None:
                text_config = getattr(thinker_config, "text_config", None)
                if text_config is not None:
                    max_len = getattr(text_config, "max_position_embeddings", None)

    if max_len is None:
        raise ValueError("max_position_embeddings not found in HFModelConfig!")
    return int(max_len)


class _UvicornServerAutoPort(uvicorn.Server):
    """Uvicorn Server that reports the system-assigned port when port=0."""

    def __init__(self, config: uvicorn.Config) -> None:
        super().__init__(config)
        self.actual_port: int | None = None
        self._startup_done: asyncio.Event = asyncio.Event()

    async def startup(self, sockets=None) -> None:
        try:
            await super().startup(sockets=sockets)
            if self.servers and self.config.port == 0:
                sock = self.servers[0].sockets[0]
                self.actual_port = sock.getsockname()[1]
            else:
                self.actual_port = self.config.port
        finally:
            self._startup_done.set()

    async def get_port(self) -> int | None:
        await self._startup_done.wait()
        return self.actual_port


async def run_uvicorn(app: FastAPI, server_args, server_address) -> tuple[int, asyncio.Task]:
    app.server_args = server_args
    config = uvicorn.Config(app, host=server_address, port=0, log_level="warning")
    server = _UvicornServerAutoPort(config)
    server_task = asyncio.create_task(server.serve())
    server_port = await server.get_port()
    if server_port is None:
        # server.startup() failed. await the task to re-raise exception from server.serve()
        await server_task

        # Fails on unexpected situation.
        raise RuntimeError("Unexpected: HTTP server started without reporting listened port")
    logger.info(f"HTTP server started on port {server_port}")
    return server_port, server_task


async def ensure_async_iterator(iterable):
    """Convert an iterable to an async iterator."""
    if hasattr(iterable, "__aiter__"):
        async for item in iterable:
            yield item
    else:
        for item in iterable:
            yield item


def _has_qwen2_vl_image_processor(processor) -> bool:
    return (
        processor is not None
        and hasattr(processor, "image_processor")
        and "Qwen2VLImageProcessor" in processor.image_processor.__class__.__name__
    )


def _collapse_consecutive_ids(prompt_ids: list[int], placeholder_ids: set[int]) -> list[int]:
    """Collapse each consecutive run of ``placeholder_ids`` down to one token.

    Boundary tokens (``<|vision_start|>``, ``<|audio_end|>``, etc.) stay in
    place because they are not in ``placeholder_ids``, so they act as
    separators between independent runs.
    """
    if not placeholder_ids:
        return prompt_ids

    arr = np.asarray(prompt_ids)
    if arr.size == 0:
        return prompt_ids

    is_placeholder = np.zeros(arr.size, dtype=bool)
    for token_id in placeholder_ids:
        is_placeholder |= arr == token_id

    mask = np.ones(arr.size, dtype=bool)
    mask[1:] &= ~(is_placeholder[1:] & is_placeholder[:-1])
    return arr[mask].tolist()


def dedup_mm_placeholder_tokens(prompt_ids: list[int], processor) -> list[int]:
    """Collapse runs of multimodal placeholder tokens before sending to vLLM.

    vLLM's multimodal prompt-replacement mechanism finds the *first* target
    placeholder token in ``prompt_token_ids`` and replaces it with the full
    expansion for that modality (e.g. N audio tokens). Any additional,
    pre-expanded placeholder tokens that the training-side processor already
    emitted would otherwise be left untouched, producing a doubly-expanded
    sequence of length roughly ``2N`` and silently decoupling the rollout
    distribution from the training distribution — see
    ``scripts/DIAGNOSIS_ROLLOUT_ACTOR_DIVERGENCE.md`` for the walk-through.

    Example (Qwen2.5-VL / Qwen3-Omni image run):

    ```
    <|vision_start|><|image_pad|><|image_pad|>...<|image_pad|><|vision_end|>
    =>
    <|vision_start|><|image_pad|><|vision_end|>
    ```

    For Qwen3-Omni the same collapse is also applied to the ``<|audio_pad|>``
    run so audio and image rollouts stay aligned with the transformers
    forward used by the actor / critic.

    If the processor is not one of the supported families, the input is
    returned unchanged.
    """
    if processor is None:
        return prompt_ids

    placeholder_ids: set[int] = set()

    if is_qwen3_omni_processor(processor):
        for token_name in ("image", "video", "audio"):
            token_id = get_processor_token_id(processor, token_name)
            if token_id is not None:
                placeholder_ids.add(token_id)
    elif _has_qwen2_vl_image_processor(processor):
        image_token_id = get_processor_token_id(processor, "image")
        video_token_id = get_processor_token_id(processor, "video")
        if image_token_id is None or video_token_id is None:
            return prompt_ids
        placeholder_ids.update({image_token_id, video_token_id})
    else:
        return prompt_ids

    return _collapse_consecutive_ids(prompt_ids, placeholder_ids)


def qwen2_5_vl_dedup_image_tokens(prompt_ids: list[int], processor):
    """Backwards-compatible alias for :func:`dedup_mm_placeholder_tokens`.

    Kept so existing rollout engines (vLLM / TRT-LLM async servers) continue
    to compile without import churn; new code should call
    ``dedup_mm_placeholder_tokens`` directly.
    """
    return dedup_mm_placeholder_tokens(prompt_ids, processor)
