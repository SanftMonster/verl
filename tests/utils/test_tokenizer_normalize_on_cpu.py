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

import numpy as np
import pytest

from verl.utils.tokenizer import (
    ensure_qwen3_omni_processor_attrs,
    get_processor_token_id,
    normalize_token_ids,
    sync_chat_template,
)


class DummyBatchEncoding:
    def __init__(self, input_ids):
        self.input_ids = input_ids


class DummyToList:
    def __init__(self, data):
        self._data = data

    def tolist(self):
        return self._data


@pytest.mark.parametrize(
    ("tokenized_output", "expected"),
    [
        # transformers v4-style direct token ids
        ([1, 2, 3], [1, 2, 3]),
        ((1, 2, 3), [1, 2, 3]),
        # common list-like outputs with tolist()/ndarray paths
        (DummyToList([1, 2, 3]), [1, 2, 3]),
        (np.array([1, 2, 3], dtype=np.int64), [1, 2, 3]),
        # transformers v5-like mapping / BatchEncoding-style outputs
        ({"input_ids": [1, 2, 3]}, [1, 2, 3]),
        ({"input_ids": DummyToList([1, 2, 3])}, [1, 2, 3]),
        ({"input_ids": [[1, 2, 3]]}, [1, 2, 3]),
        (DummyBatchEncoding([1, 2, 3]), [1, 2, 3]),
        (DummyBatchEncoding(DummyToList([[1, 2, 3]])), [1, 2, 3]),
        # scalar item() support
        ([np.int64(1), np.int32(2), np.int16(3)], [1, 2, 3]),
    ],
)
def test_normalize_token_ids_valid_outputs(tokenized_output, expected):
    assert normalize_token_ids(tokenized_output) == expected


@pytest.mark.parametrize(
    "tokenized_output",
    [
        "not-token-ids",
        {"attention_mask": [1, 1, 1]},
        [[1, 2], [3, 4]],  # ambiguous batched ids should fail fast
        [1, object(), 3],
    ],
)
def test_normalize_token_ids_invalid_outputs(tokenized_output):
    with pytest.raises(TypeError):
        normalize_token_ids(tokenized_output)


class DummyChatTemplateHolder:
    def __init__(self, chat_template=None):
        self.chat_template = chat_template


class DummyTokenizer:
    def __init__(self, token_to_id):
        self.token_to_id = token_to_id

    def convert_tokens_to_ids(self, token):
        return self.token_to_id[token]


class DummyImageProcessor:
    def __init__(self, merge_size):
        self.merge_size = merge_size


def test_sync_chat_template_copies_from_processor_to_tokenizer():
    tokenizer = DummyChatTemplateHolder()
    processor = DummyChatTemplateHolder(chat_template="processor-template")

    sync_chat_template(tokenizer, processor)

    assert tokenizer.chat_template == "processor-template"
    assert processor.chat_template == "processor-template"


def test_sync_chat_template_custom_template_overrides_both():
    tokenizer = DummyChatTemplateHolder(chat_template="tokenizer-template")
    processor = DummyChatTemplateHolder(chat_template="processor-template")

    sync_chat_template(tokenizer, processor, custom_chat_template="custom-template")

    assert tokenizer.chat_template == "custom-template"
    assert processor.chat_template == "custom-template"


def test_get_processor_token_id_falls_back_to_token_string():
    processor = DummyChatTemplateHolder()
    processor.image_token = "<|image_pad|>"
    processor.tokenizer = DummyTokenizer({"<|image_pad|>": 151655})

    assert get_processor_token_id(processor, "image") == 151655


def test_ensure_qwen3_omni_processor_attrs_populates_missing_fields():
    class Qwen3OmniMoeProcessor:
        def __init__(self):
            self.config = DummyChatTemplateHolder()
            self.config.im_start_token_id = 151652
            self.config.im_end_token_id = 151653
            self.config.thinker_config = DummyChatTemplateHolder()
            self.config.thinker_config.audio_token_id = 151646
            self.config.thinker_config.image_token_id = 151655
            self.config.thinker_config.video_token_id = 151656
            self.config.thinker_config.position_id_per_seconds = 25
            self.config.thinker_config.audio_start_token_id = 151647
            self.config.thinker_config.vision_config = DummyChatTemplateHolder()
            self.config.thinker_config.vision_config.spatial_merge_size = 2
            self.config.talker_config = DummyChatTemplateHolder()
            self.config.talker_config.audio_start_token_id = 151669
            self.config.talker_config.vision_start_token_id = 151652
            self.config.talker_config.position_id_per_seconds = 25
            self.image_processor = DummyImageProcessor(merge_size=2)
            self.image_token = "<|image_pad|>"
            self.video_token = "<|video_pad|>"
            self.audio_token = "<|audio_pad|>"
            self.tokenizer = DummyTokenizer(
                {
                    "<|image_pad|>": 151655,
                    "<|video_pad|>": 151656,
                    "<|audio_pad|>": 151646,
                }
            )

    processor = Qwen3OmniMoeProcessor()
    ensure_qwen3_omni_processor_attrs(processor)

    assert processor.image_token_id == 151655
    assert processor.video_token_id == 151656
    assert processor.audio_token_id == 151646
    assert processor.audio_start_token_id == 151669
    assert processor.audio_end_token_id == 151670
    assert processor.position_id_per_seconds == 25
    assert processor.spatial_merge_size == 2
    assert processor.config.image_token_id == 151655
    assert processor.config.video_token_id == 151656
    assert processor.config.audio_token_id == 151646
    assert processor.config.audio_start_token_id == 151669
    assert processor.config.audio_end_token_id == 151670
    assert processor.config.position_id_per_seconds == 25
    assert processor.vision_start_token_id == 151652
    assert processor.vision_end_token_id == 151653
    assert processor.config.vision_start_token_id == 151652
    assert processor.config.vision_end_token_id == 151653
