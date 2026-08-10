# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import random

from benchmarks.cache.workload import _sample_prompt


class DiscontinuousLengthTokenizer:
    """Expose a narrow valid length that proportional search can skip."""

    all_special_ids: list[int] = []
    vocab_size = 10000

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return " ".join(str(token_id) for token_id in token_ids)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        source = [int(part) for part in text.split()] if text else []
        source_length = len(source)

        if source_length == 1016:
            encoded_length = 1024
        elif source_length >= 1010:
            encoded_length = source_length + 32
        else:
            encoded_length = max(0, source_length - 16)

        if encoded_length <= source_length:
            return source[:encoded_length]
        return source + [self.vocab_size - 1] * (encoded_length - source_length)


def test_search_recovers_when_proportional_steps_skip_valid_length() -> None:
    tokenizer = DiscontinuousLengthTokenizer()
    prompt, encoded = _sample_prompt(
        rng=random.Random(1),
        tokenizer=tokenizer,
        allowed_tokens=tuple(range(tokenizer.vocab_size - 1)),
        requested_length=1024,
        tolerance=2,
    )

    assert prompt
    assert abs(len(encoded) - 1024) <= 2
