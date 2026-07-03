"""Tests for the compression pipeline orchestrator: budget trigger, no-op
fast path, active-segment protection, and the skeletonize technique wired
end-to-end through segmenter -> skeletonizer."""
import pytest

from coordinator.context_compression.config import CompressionConfig
from coordinator.context_compression.pipeline import Message, compress_messages, compress_prompt


def _config(**overrides: object) -> CompressionConfig:
    base = dict(
        enabled=True,
        token_budget=50,
        safety_margin=0.0,
        active_segments=1,
        techniques=("skeletonize",),
        summarizer_model="qwen2.5-0.5b-gguf",
        summarizer_max_tokens=256,
        llmlingua_model="unused",
        llmlingua_rate=0.5,
    )
    base.update(overrides)
    return CompressionConfig(**base)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_disabled_is_a_pure_noop() -> None:
    messages = [Message(role="user", content="x" * 10_000)]
    outcome = await compress_messages(messages, _config(enabled=False))
    assert outcome.messages == messages
    assert outcome.skipped_reason == "disabled"
    assert outcome.applied == ()


@pytest.mark.asyncio
async def test_under_budget_is_a_pure_noop() -> None:
    messages = [Message(role="user", content="short")]
    outcome = await compress_messages(messages, _config(token_budget=8192))
    assert outcome.messages == messages
    assert outcome.skipped_reason == "under_budget"
    assert outcome.applied == ()


@pytest.mark.asyncio
async def test_over_budget_skeletonizes_peripheral_code_only() -> None:
    old_code = "```python\ndef old_helper(x, y):\n    z = x + y\n    return z\n```\n"
    active_code = "```python\ndef current_edit(a):\n    return a * 2\n```\n"
    messages = [
        Message(role="user", content=old_code * 3),  # peripheral: large, older
        Message(role="user", content=active_code),  # active: the current turn
    ]
    outcome = await compress_messages(messages, _config(token_budget=5))
    assert "skeletonize" in outcome.applied
    # Active segment (last code block) must stay verbatim.
    assert "return a * 2" in outcome.messages[-1].content
    # Peripheral occurrences must have been skeletonized.
    assert "z = x + y" not in outcome.messages[0].content
    assert "def old_helper(x, y):" in outcome.messages[0].content  # signature kept
    assert outcome.tokens_after < outcome.tokens_before


@pytest.mark.asyncio
async def test_unknown_technique_name_is_skipped_not_fatal() -> None:
    """A technique enabled in config but not yet registered (e.g. 'summarize'
    before Task 9 ships) must not crash the pipeline."""
    messages = [Message(role="user", content="```python\n" + "x = 1\n" * 200 + "```\n")]
    outcome = await compress_messages(messages, _config(techniques=("summarize",)))
    assert outcome.applied == ()  # nothing registered for "summarize" yet
    assert outcome.messages[0].content == messages[0].content


@pytest.mark.asyncio
async def test_compress_prompt_wraps_a_single_string() -> None:
    prompt = "```python\n" + "def f():\n    pass\n" * 50 + "```\n"
    outcome = await compress_prompt(prompt, _config(token_budget=5))
    assert isinstance(outcome.messages[-1].content, str)
    assert outcome.tokens_after <= outcome.tokens_before


@pytest.mark.asyncio
async def test_stops_early_once_back_under_budget() -> None:
    """Only one technique is registered in Phase 1, so this mostly proves the
    loop terminates and reports exactly the techniques it actually ran."""
    messages = [Message(role="user", content="```python\ndef f():\n    return 1\n```\n")]
    outcome = await compress_messages(messages, _config(token_budget=1))
    assert set(outcome.applied) <= {"skeletonize"}
