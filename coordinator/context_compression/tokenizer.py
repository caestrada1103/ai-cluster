"""Coarse, offline token-count estimate used for the compression budget check.

Deliberately NOT tied to any specific model's real tokenizer — see
pending-work/12-context-compression-pipeline.md's "Honest risks" section for
the full rationale. A character-density heuristic, always compared against
the budget after `apply_safety_margin()` inflates it, biasing toward
over-compressing rather than under-compressing.
"""
from __future__ import annotations

import math

# Not model-specific: common BPE vocabularies (GPT-2/GPT-4/LLaMA-family
# tokenizers) average close to 4 characters/token for English prose. Code
# runs denser — punctuation, operators, and identifiers split into more
# subword pieces — so it gets its own, lower ratio.
_CHARS_PER_TOKEN_NL = 4.0
_CHARS_PER_TOKEN_CODE = 3.0


def estimate_tokens(text: str, *, is_code: bool = False) -> int:
    """Rough token count for `text`. Zero only for empty input."""
    if not text:
        return 0
    chars_per_token = _CHARS_PER_TOKEN_CODE if is_code else _CHARS_PER_TOKEN_NL
    return max(1, math.ceil(len(text) / chars_per_token))


def apply_safety_margin(token_estimate: int, margin: float) -> int:
    """Inflate a raw estimate by `margin` (e.g. 0.20 == +20%) before it is
    compared against `CompressionConfig.token_budget`. See the module
    docstring — this is the tokenizer-parity hedge."""
    return math.ceil(token_estimate * (1.0 + margin))
