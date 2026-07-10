"""Tests for the offline token-count heuristic (no model, no network)."""
from coordinator.context_compression.tokenizer import apply_safety_margin, estimate_tokens


def test_estimate_tokens_empty_string_is_zero() -> None:
    assert estimate_tokens("") == 0


def test_estimate_tokens_nl_uses_four_chars_per_token() -> None:
    assert estimate_tokens("a" * 40) == 10  # 40 / 4.0


def test_estimate_tokens_code_uses_three_chars_per_token() -> None:
    assert estimate_tokens("a" * 30, is_code=True) == 10  # 30 / 3.0


def test_estimate_tokens_never_zero_for_nonempty_text() -> None:
    assert estimate_tokens("a") == 1
    assert estimate_tokens("a", is_code=True) == 1


def test_estimate_tokens_rounds_up() -> None:
    assert estimate_tokens("a" * 5) == 2  # ceil(5/4) == 2, not 1


def test_apply_safety_margin_inflates() -> None:
    assert apply_safety_margin(100, 0.20) == 120
    assert apply_safety_margin(0, 0.20) == 0


def test_apply_safety_margin_zero_margin_is_identity() -> None:
    assert apply_safety_margin(137, 0.0) == 137
