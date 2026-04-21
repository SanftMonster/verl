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
"""Utils for tokenization."""

import types
import warnings

__all__ = [
    "hf_tokenizer",
    "hf_processor",
    "normalize_token_ids",
    "is_qwen3_omni_processor",
    "build_multimodal_processor_inputs",
    "ensure_qwen3_omni_processor_attrs",
    "get_processor_token_id",
    "sync_chat_template",
]


def normalize_token_ids(tokenized_output) -> list[int]:
    """Normalize tokenizer outputs into a flat ``list[int]``.

    This handles Transformers 4/5 differences where ``apply_chat_template(tokenize=True)``
    may return either ``list[int]`` or a ``BatchEncoding``/mapping with ``input_ids``.
    """

    token_ids = tokenized_output
    if isinstance(tokenized_output, dict):
        if "input_ids" in tokenized_output:
            token_ids = tokenized_output["input_ids"]
    elif hasattr(tokenized_output, "input_ids"):
        token_ids = tokenized_output.input_ids

    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()

    if isinstance(token_ids, tuple):
        token_ids = list(token_ids)

    if isinstance(token_ids, list) and len(token_ids) == 1 and isinstance(token_ids[0], list | tuple):
        token_ids = list(token_ids[0])

    if not isinstance(token_ids, list):
        raise TypeError(f"token_ids must be list-like token ids, got {type(token_ids).__name__}: {token_ids!r}")

    normalized_ids = []
    for idx, token_id in enumerate(token_ids):
        if hasattr(token_id, "item"):
            token_id = token_id.item()
        try:
            normalized_ids.append(int(token_id))
        except (TypeError, ValueError) as e:
            raise TypeError(f"token_id must be int-convertible, got {type(token_id).__name__}: {token_id!r}") from e
    return normalized_ids


def is_qwen3_omni_processor(processor) -> bool:
    """Return True when ``processor`` is a Qwen3-Omni processor."""
    return processor is not None and processor.__class__.__name__ == "Qwen3OmniMoeProcessor"


def get_processor_token_id(processor, token_name: str) -> int | None:
    """Resolve a multimodal special token id from a processor.

    Newer processors may expose ``image_token``/``video_token`` strings instead
    of ``image_token_id``/``video_token_id`` integers. Fall back to tokenizer
    conversion so rollout code can stay processor-agnostic.
    """

    if processor is None:
        return None

    token_id_attr = f"{token_name}_token_id"
    token_id = getattr(processor, token_id_attr, None)
    if token_id is not None:
        return int(token_id)

    token_attr = f"{token_name}_token"
    token = getattr(processor, token_attr, None)
    tokenizer = getattr(processor, "tokenizer", None)
    if token is not None and tokenizer is not None:
        converted = tokenizer.convert_tokens_to_ids(token)
        if converted is not None:
            return int(converted)

    return None


def ensure_qwen3_omni_processor_attrs(processor):
    """Populate processor attributes expected by Qwen3-Omni helper methods.

    ``Qwen3OmniMoeProcessor`` exposes some multimodal metadata on nested config
    or token-string fields rather than direct ``*_token_id`` attributes. The
    bound ``get_rope_index`` implementation expects these attributes on the
    processor instance, so mirror them when missing.
    """

    if not is_qwen3_omni_processor(processor):
        return processor

    config = getattr(processor, "config", None)
    thinker_config = getattr(config, "thinker_config", None)
    talker_config = getattr(config, "talker_config", None)
    vision_config = getattr(thinker_config, "vision_config", None) or getattr(config, "vision_config", None)
    default_token_ids = {
        "audio_token_id": 151646,
        "image_token_id": 151655,
        "video_token_id": 151656,
        "vision_start_token_id": 151652,
        "vision_end_token_id": 151653,
        "audio_start_token_id": 151669,
        "audio_end_token_id": 151670,
        "position_id_per_seconds": 25,
    }

    def _get_config_attr(*names, sources=None):
        search_sources = sources or (config, thinker_config, talker_config)
        for source in search_sources:
            if source is None:
                continue
            for name in names:
                value = getattr(source, name, None)
                if value is not None:
                    return value
        return None

    def _set_if_missing(attr_name: str, value):
        if value is not None and not hasattr(processor, attr_name):
            setattr(processor, attr_name, value)

    def _set_config_if_missing(attr_name: str, value):
        if config is not None and value is not None and not hasattr(config, attr_name):
            setattr(config, attr_name, value)

    _set_if_missing(
        "image_token_id",
        _get_config_attr("image_token_id", sources=(config, thinker_config, talker_config))
        or default_token_ids["image_token_id"],
    )
    _set_if_missing(
        "video_token_id",
        _get_config_attr("video_token_id", sources=(config, thinker_config, talker_config))
        or default_token_ids["video_token_id"],
    )
    _set_if_missing(
        "audio_token_id",
        _get_config_attr("audio_token_id", sources=(config, thinker_config, talker_config))
        or default_token_ids["audio_token_id"],
    )
    _set_if_missing(
        "vision_start_token_id",
        _get_config_attr("vision_start_token_id", "im_start_token_id", sources=(config, talker_config, thinker_config))
        or default_token_ids["vision_start_token_id"],
    )
    _set_if_missing(
        "vision_end_token_id",
        _get_config_attr("vision_end_token_id", "im_end_token_id", sources=(config, talker_config, thinker_config))
        or default_token_ids["vision_end_token_id"],
    )
    _set_if_missing(
        "audio_start_token_id",
        _get_config_attr("audio_start_token_id", sources=(config, talker_config, thinker_config))
        or default_token_ids["audio_start_token_id"],
    )
    _set_if_missing(
        "audio_end_token_id",
        _get_config_attr("audio_end_token_id", sources=(config, talker_config, thinker_config))
        or default_token_ids["audio_end_token_id"],
    )
    _set_if_missing(
        "position_id_per_seconds",
        _get_config_attr("position_id_per_seconds", sources=(config, talker_config, thinker_config))
        or default_token_ids["position_id_per_seconds"],
    )
    _set_if_missing("spatial_merge_size", getattr(vision_config, "spatial_merge_size", None))
    _set_if_missing("tokens_per_second", getattr(vision_config, "tokens_per_second", None))

    if not hasattr(processor, "spatial_merge_size"):
        merge_size = getattr(getattr(processor, "image_processor", None), "merge_size", None)
        _set_if_missing("spatial_merge_size", merge_size)

    for token_name in ("image", "video", "audio"):
        token_id_attr = f"{token_name}_token_id"
        if not hasattr(processor, token_id_attr):
            token_id = get_processor_token_id(processor, token_name)
            _set_if_missing(token_id_attr, token_id or default_token_ids.get(token_id_attr))

    for attr_name in (
        "image_token_id",
        "video_token_id",
        "audio_token_id",
        "vision_start_token_id",
        "vision_end_token_id",
        "audio_start_token_id",
        "audio_end_token_id",
        "position_id_per_seconds",
    ):
        _set_config_if_missing(attr_name, getattr(processor, attr_name, None))

    return processor


def _split_videos_and_metadata(videos):
    if videos is None:
        return None, None
    videos = list(videos)
    if len(videos) > 0 and isinstance(videos[0], tuple):
        video_values, video_metadata = zip(*videos, strict=False)
        return list(video_values), list(video_metadata)
    return videos, None


def build_multimodal_processor_inputs(
    processor,
    *,
    text,
    images=None,
    videos=None,
    audio=None,
    mm_processor_kwargs=None,
    return_tensors: str = "pt",
):
    """Build kwargs for multimodal processor calls.

    This keeps the existing VL flow intact while extending it with audio-aware
    paths needed by Qwen3-Omni and generic audio-input processors.
    """
    processor_kwargs = dict(mm_processor_kwargs or {})
    if audio is not None and "sampling_rate" not in processor_kwargs:
        sampling_rate = getattr(getattr(processor, "feature_extractor", None), "sampling_rate", None)
        if sampling_rate is not None:
            processor_kwargs["sampling_rate"] = int(sampling_rate)

    videos, video_metadata = _split_videos_and_metadata(videos)
    processor_kwargs.setdefault("return_tensors", return_tensors)

    if is_qwen3_omni_processor(processor):
        return processor(text=text, images=images, videos=videos, audio=audio, **processor_kwargs)

    if video_metadata is not None:
        processor_kwargs.setdefault("video_metadata", video_metadata)
        processor_kwargs.setdefault("do_sample_frames", False)

    return processor(text=text, images=images, videos=videos, **processor_kwargs)


def set_pad_token_id(tokenizer):
    """Set pad_token_id to eos_token_id if it is None.

    Args:
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to be set.

    """
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        warnings.warn(f"tokenizer.pad_token_id is None. Now set to {tokenizer.eos_token_id}", stacklevel=1)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        warnings.warn(f"tokenizer.pad_token is None. Now set to {tokenizer.eos_token}", stacklevel=1)


def sync_chat_template(tokenizer, processor=None, custom_chat_template=None):
    """Keep tokenizer and processor chat templates consistent.

    Some multimodal processors expose a chat template while the paired tokenizer
    does not. Agent-loop initialization uses the tokenizer path for system-prompt
    detection, so we mirror the template to whichever side is missing.
    """

    if tokenizer is None and processor is None:
        return

    if custom_chat_template is not None:
        if tokenizer is not None:
            tokenizer.chat_template = custom_chat_template
        if processor is not None:
            processor.chat_template = custom_chat_template
        return

    tokenizer_template = getattr(tokenizer, "chat_template", None) if tokenizer is not None else None
    processor_template = getattr(processor, "chat_template", None) if processor is not None else None

    if tokenizer is not None and not tokenizer_template and processor_template:
        tokenizer.chat_template = processor_template
    elif processor is not None and not processor_template and tokenizer_template:
        processor.chat_template = tokenizer_template


def hf_tokenizer(name_or_path, correct_pad_token=True, correct_gemma2=True, **kwargs):
    """Create a huggingface pretrained tokenizer which correctness handles eos and pad tokens.

    Args:

        name (str): The name of the tokenizer.
        correct_pad_token (bool): Whether to correct the pad token id.
        correct_gemma2 (bool): Whether to correct the gemma2 tokenizer.

    Returns:

        transformers.PreTrainedTokenizer: The pretrained tokenizer.

    """
    from transformers import AutoTokenizer

    if correct_gemma2 and isinstance(name_or_path, str) and "gemma-2-2b-it" in name_or_path:
        # the EOS token in gemma2 is ambiguious, which may worsen RL performance.
        # https://huggingface.co/google/gemma-2-2b-it/commit/17a01657f5c87135bcdd0ec7abb4b2dece04408a
        warnings.warn(
            "Found gemma-2-2b-it tokenizer. Set eos_token and eos_token_id to <end_of_turn> and 107.", stacklevel=1
        )
        kwargs["eos_token"] = "<end_of_turn>"
        kwargs["eos_token_id"] = 107
    tokenizer = AutoTokenizer.from_pretrained(name_or_path, **kwargs)
    if correct_pad_token:
        set_pad_token_id(tokenizer)
    return tokenizer


def hf_processor(name_or_path, **kwargs):
    """Create a huggingface processor to process multimodal data.

    Args:
        name_or_path (str): The name of the processor.

    Returns:
        Optional[transformers.ProcessorMixin]: The pretrained multimodal processor.
        Returns ``None`` for text-only models (including AutoProcessor fallbacks to
        tokenizer backends such as ``TokenizersBackend``).
    """
    from transformers import AutoConfig, AutoProcessor, PreTrainedTokenizerBase

    try:
        processor = AutoProcessor.from_pretrained(name_or_path, **kwargs)
        # In newer transformers, AutoProcessor may legitimately fall back to a
        # tokenizer backend (e.g. TokenizersBackend) for text-only models.
        # Treat it as "no multimodal processor" and let callers use hf_tokenizer.
        if isinstance(processor, PreTrainedTokenizerBase):
            return None

        config = AutoConfig.from_pretrained(name_or_path, **kwargs)

        # Bind vlm model's get_rope_index method to processor
        processor.config = config
        model_class = None
        match processor.__class__.__name__:
            case "Qwen2VLProcessor":
                from transformers.models.qwen2_vl import Qwen2VLModel

                model_class = Qwen2VLModel
            case "Qwen2_5_VLProcessor":
                from transformers.models.qwen2_5_vl import Qwen2_5_VLModel

                model_class = Qwen2_5_VLModel
            case "Qwen3VLProcessor":
                from transformers.models.qwen3_vl import Qwen3VLModel

                model_class = Qwen3VLModel
            case "Qwen3OmniMoeProcessor":
                from transformers.models.qwen3_omni_moe import Qwen3OmniMoeThinkerForConditionalGeneration

                model_class = Qwen3OmniMoeThinkerForConditionalGeneration
            case "Glm4vImageProcessor":
                from transformers.models.glm4v import Glm4vModel

                model_class = Glm4vModel
            case "MllamaProcessor":
                pass  # MllamaProcessor and MllamaModel doesn't have get_rope_index property
            case _:
                raise ValueError(f"Unsupported processor type: {processor.__class__.__name__}")

        if model_class is not None:
            processor.get_rope_index = types.MethodType(model_class.get_rope_index, processor)
            if hasattr(model_class, "get_vision_position_ids"):
                processor.get_vision_position_ids = types.MethodType(model_class.get_vision_position_ids, processor)
            ensure_qwen3_omni_processor_attrs(processor)
    except Exception as e:
        processor = None
        # TODO(haibin.lin): try-catch should be removed after adding transformer version req to setup.py to avoid
        # silent failure
        warnings.warn(f"Failed to create processor: {e}. This may affect multimodal processing", stacklevel=1)
    # Avoid load tokenizer, see:
    # https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/models/auto/processing_auto.py#L344
    if processor is not None and "Processor" not in processor.__class__.__name__:
        processor = None
    return processor
