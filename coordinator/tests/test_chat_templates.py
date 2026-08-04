"""Tests for the in-process chat-template selection (coordinator/api.py, item 4).

Before this fix, `create_chat_completion` flattened every non-llamaserver
model's chat history with one hardcoded Zephyr-style template regardless of
`family`. Verified on hardware with a Qwen model: the reply terminated
correctly and then emitted a spurious `<|user|>` turn replaying the prompt,
because Qwen was never trained on Zephyr's `</s>`-separated turn format and
the coordinator had no stop sequence to cut the reply at the right point.

These tests cover:
  - `_select_chat_template` picks a family-specific builder for every
    `ModelFamily` value, and falls back (with stop sequences) for unregistered
    models.
  - Each builder produces the expected role markers for its family.
  - `_truncate_at_stop` — the safety net applied regardless of which template
    produced the prompt.
  - End-to-end through `create_chat_completion`: a Qwen-family model's output
    with a spurious replayed turn comes back truncated.
"""
from types import SimpleNamespace
from typing import Any, Dict

import pytest

from coordinator.api import (
    _FAMILY_CHAT_TEMPLATES,
    _ZEPHYR_FALLBACK_STOP,
    ChatCompletionRequest,
    ChatMessage,
    _build_chatml_prompt,
    _build_deepseek_prompt,
    _build_gemma_prompt,
    _build_llama3_prompt,
    _build_mistral_prompt,
    _build_phi_prompt,
    _build_zephyr_prompt,
    _select_chat_template,
    _truncate_at_stop,
    create_chat_completion,
)
from coordinator.models import ModelConfig, ModelFamily
from coordinator.tests.conftest import make_settings

_MESSAGES = [
    ChatMessage(role="system", content="Be terse."),
    ChatMessage(role="user", content="hi there"),
]


def _make_cfg(name: str, family: ModelFamily, engine: str = "llamacpp") -> ModelConfig:
    return ModelConfig(
        name=name,
        family=family,
        parameters="1B",
        min_memory_gb=1,
        recommended_gpus=1,
        max_gpus=1,
        num_layers=1,
        hidden_size=1,
        num_attention_heads=1,
        vocab_size=1,
        max_seq_len=4096,
        intermediate_size=1,
        engine=engine,
        gguf_repo_id="org/repo" if engine != "burn" else None,
        gguf_file="model.gguf" if engine != "burn" else None,
    )


# ---------------------------------------------------------------------------
# _select_chat_template
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", list(ModelFamily))
def test_select_chat_template_covers_every_family(family: ModelFamily) -> None:
    """Every ModelFamily value must resolve to a dedicated (non-fallback) template."""
    cfg = _make_cfg(f"model-{family.value}", family)
    builder, stop = _select_chat_template(cfg, cfg.name)
    assert builder is not _build_zephyr_prompt
    assert builder is _FAMILY_CHAT_TEMPLATES[family.value][0]
    assert stop == _FAMILY_CHAT_TEMPLATES[family.value][1]


def test_select_chat_template_falls_back_for_unregistered_model(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """No registry entry (model_cfg=None) -> generic template + stop sequences,
    with a warning pointing at engine="llamaserver"."""
    with caplog.at_level("WARNING"):
        builder, stop = _select_chat_template(None, "totally-unknown-model")
    assert builder is _build_zephyr_prompt
    assert stop == _ZEPHYR_FALLBACK_STOP
    assert any("llamaserver" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# Individual template builders
# ---------------------------------------------------------------------------


def test_chatml_prompt_uses_im_start_end() -> None:
    prompt = _build_chatml_prompt(_MESSAGES)
    assert "<|im_start|>system\nBe terse.<|im_end|>\n" in prompt
    assert "<|im_start|>user\nhi there<|im_end|>\n" in prompt
    assert prompt.endswith("<|im_start|>assistant\n")
    # NOT the old hardcoded template's markers.
    assert "<|system|>" not in prompt


def test_llama3_prompt_uses_header_blocks() -> None:
    prompt = _build_llama3_prompt(_MESSAGES)
    assert prompt.startswith("<|begin_of_text|>")
    assert "<|start_header_id|>system<|end_header_id|>\n\nBe terse.<|eot_id|>" in prompt
    assert prompt.endswith("<|start_header_id|>assistant<|end_header_id|>\n\n")


def test_mistral_prompt_wraps_user_in_inst() -> None:
    prompt = _build_mistral_prompt(_MESSAGES)
    assert "[INST] Be terse.\n\nhi there [/INST]" in prompt


def test_gemma_prompt_uses_start_end_of_turn() -> None:
    prompt = _build_gemma_prompt(_MESSAGES)
    assert "<start_of_turn>user\nBe terse.\n\nhi there<end_of_turn>\n" in prompt
    assert prompt.endswith("<start_of_turn>model\n")


def test_phi_prompt_uses_end_token_not_zephyr_eos() -> None:
    prompt = _build_phi_prompt(_MESSAGES)
    assert "<|system|>\nBe terse.<|end|>\n" in prompt
    assert "</s>" not in prompt  # distinguishes it from the old Zephyr fallback


def test_deepseek_prompt_uses_user_assistant_markers() -> None:
    prompt = _build_deepseek_prompt(_MESSAGES)
    assert "<｜User｜>hi there" in prompt
    assert prompt.endswith("<｜Assistant｜>")


def test_zephyr_fallback_unchanged_shape() -> None:
    """The legacy fallback template's shape is preserved for models that still
    rely on it (family absent from the registry)."""
    prompt = _build_zephyr_prompt(_MESSAGES)
    assert "<|system|>\nBe terse.</s>\n" in prompt
    assert "<|user|>\nhi there</s>\n" in prompt
    assert prompt.endswith("<|assistant|>\n")


# ---------------------------------------------------------------------------
# _truncate_at_stop
# ---------------------------------------------------------------------------


def test_truncate_at_stop_cuts_at_earliest_marker() -> None:
    text = "The answer is 42.<|im_start|>user\nreplay of the prompt<|im_end|>"
    assert _truncate_at_stop(text, ["<|im_end|>", "<|im_start|>"]) == "The answer is 42."


def test_truncate_at_stop_noop_when_absent() -> None:
    text = "clean reply, no spurious turn"
    assert _truncate_at_stop(text, ["<|im_end|>", "<|im_start|>"]) == text


def test_truncate_at_stop_empty_stop_list() -> None:
    text = "clean reply"
    assert _truncate_at_stop(text, []) == text


# ---------------------------------------------------------------------------
# End-to-end through create_chat_completion
# ---------------------------------------------------------------------------


class _FakeCoordinator:
    def __init__(self, settings: Any, reply_text: str) -> None:
        self.settings = settings
        self.active_requests: Dict[str, Any] = {}
        self.last_prompt: str = ""
        self._reply_text = reply_text

    async def infer(self, model_name: str, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        self.last_prompt = prompt
        return {
            "request_id": "r1",
            "text": self._reply_text,
            "tokens_generated": 42,
            "processing_time_ms": 1.0,
            "worker_id": "w1",
        }


def _fake_request(coordinator: _FakeCoordinator, body: Any) -> Any:
    raw: bytes = body.model_dump_json().encode()

    async def _body() -> bytes:
        return raw

    app_state = SimpleNamespace(coordinator=coordinator)
    app = SimpleNamespace(state=app_state)
    return SimpleNamespace(app=app, body=_body)


@pytest.mark.asyncio
async def test_chat_completion_uses_qwen_template_and_truncates_spurious_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end reproduction of the hardware bug: a Qwen-family model's raw
    reply contains a spurious replayed turn after the real answer; the
    endpoint must (a) prompt with ChatML, not Zephyr, and (b) return only the
    real answer to the client."""
    from coordinator.models import ModelRegistry

    cfg = _make_cfg("qwen-test-model", ModelFamily.QWEN)
    monkeypatch.setitem(ModelRegistry.MODELS, cfg.name, cfg)

    settings = make_settings(context_compression_enabled=False)
    raw_reply = "The answer is 42.<|im_start|>user\nreplay of the prompt<|im_end|>"
    coordinator = _FakeCoordinator(settings, raw_reply)
    body = ChatCompletionRequest(
        model=cfg.name, messages=[{"role": "user", "content": "What is the answer?"}]
    )

    response = await create_chat_completion(_fake_request(coordinator, body))

    assert isinstance(response, dict)
    assert response["choices"][0]["message"]["content"] == "The answer is 42."
    assert "<|im_start|>user\nWhat is the answer?<|im_end|>" in coordinator.last_prompt
    assert "<|system|>" not in coordinator.last_prompt  # not the old hardcoded template


@pytest.mark.asyncio
async def test_chat_completion_unregistered_model_still_truncates_fallback_template() -> None:
    """Unregistered model ("m", not in the registry) keeps working (no
    template deletion) but still truncates a spurious <|user|> replay via the
    fallback's stop sequences."""
    settings = make_settings(context_compression_enabled=False)
    raw_reply = "sure, here you go.<|user|>\nreplay</s>\n"
    coordinator = _FakeCoordinator(settings, raw_reply)
    body = ChatCompletionRequest(model="m", messages=[{"role": "user", "content": "hi"}])

    response = await create_chat_completion(_fake_request(coordinator, body))

    assert isinstance(response, dict)
    assert response["choices"][0]["message"]["content"] == "sure, here you go."
