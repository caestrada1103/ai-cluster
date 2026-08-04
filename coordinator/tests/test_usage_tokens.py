"""Regression tests for `usage.prompt_tokens` on the in-process chat completion
path (coordinator/api.py::_build_flat_response).

Before the fix, `usage.prompt_tokens` was hardcoded to 0 because the gRPC
`InferenceResponse` (proto/cluster.proto) carries no prompt-token count for
the coordinator to read — only `tokens_generated` (completion tokens). The
fix estimates prompt tokens coordinator-side from the flattened prompt text
using the same coarse, model-agnostic heuristic already used by the
context-compression budget check, so `total_tokens` is internally consistent
instead of silently wrong.
"""
from types import SimpleNamespace
from typing import Any, Dict

import pytest

from coordinator.api import ChatCompletionRequest, _build_flat_response, create_chat_completion
from coordinator.context_compression.tokenizer import estimate_tokens
from coordinator.tests.conftest import make_settings


class _FakeCoordinator:
    """Mimics ClusterCoordinator.infer() just enough for the API layer."""

    def __init__(self, settings: Any, completion_tokens: int = 200) -> None:
        self.settings = settings
        self.active_requests: Dict[str, Any] = {}
        self.last_prompt: str = ""
        self._completion_tokens = completion_tokens

    async def infer(self, model_name: str, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        self.last_prompt = prompt
        return {
            "request_id": "r1",
            "text": "ok",
            "tokens_generated": self._completion_tokens,
            "processing_time_ms": 1.0,
            "worker_id": "w1",
        }


def _fake_request(coordinator: _FakeCoordinator, body: Any) -> Any:
    """Build a duck-typed Request whose .body() returns the model's raw JSON.

    Mirrors the helper in test_context_compression_api.py. Model "m" is
    unknown to the registry, so this always takes the in-process path.
    """
    raw: bytes = body.model_dump_json().encode()

    async def _body() -> bytes:
        return raw

    app_state = SimpleNamespace(coordinator=coordinator)
    app = SimpleNamespace(state=app_state)
    return SimpleNamespace(app=app, state=SimpleNamespace(), body=_body)


def test_build_flat_response_prompt_tokens_not_hardcoded_zero() -> None:
    """Direct unit test of the helper: prompt_tokens must reflect the prompt."""
    result = {"request_id": "r1", "text": "hi", "tokens_generated": 200}
    prompt = "x" * 400  # long enough that the estimate is unambiguously > 0
    resp = _build_flat_response(result, "some-model", prompt)

    assert resp["usage"]["prompt_tokens"] > 0
    assert resp["usage"]["prompt_tokens"] == estimate_tokens(prompt)
    assert resp["usage"]["completion_tokens"] == 200
    assert (
        resp["usage"]["total_tokens"]
        == resp["usage"]["prompt_tokens"] + resp["usage"]["completion_tokens"]
    )


def test_build_flat_response_empty_prompt_is_zero() -> None:
    """Sanity: an empty prompt legitimately estimates to 0 tokens, not an error."""
    result = {"request_id": "r1", "text": "hi", "tokens_generated": 5}
    resp = _build_flat_response(result, "some-model", "")
    assert resp["usage"]["prompt_tokens"] == 0
    assert resp["usage"]["total_tokens"] == 5


@pytest.mark.asyncio
async def test_chat_completion_endpoint_reports_nonzero_prompt_tokens() -> None:
    """End-to-end through create_chat_completion: mirrors the hardware repro
    (`usage: {prompt_tokens: 0, completion_tokens: 200, total_tokens: 200}`)
    and asserts it no longer happens."""
    settings = make_settings(context_compression_enabled=False)
    coordinator = _FakeCoordinator(settings, completion_tokens=200)
    body = ChatCompletionRequest(
        model="m",
        messages=[{"role": "user", "content": "Explain the theory of relativity in detail."}],
    )
    response = await create_chat_completion(_fake_request(coordinator, body))

    assert isinstance(response, dict)
    usage = response["usage"]
    assert usage["prompt_tokens"] > 0
    assert usage["completion_tokens"] == 200
    assert usage["total_tokens"] == usage["prompt_tokens"] + 200


def test_gguf_n_cpu_moe_forwarded_to_worker_metadata() -> None:
    """`gguf.n_cpu_moe` must reach the worker as an `n_cpu_moe` metadata key.

    The worker's llamaserver path supports `--n-cpu-moe` (MoE expert offload —
    the lever that fits a large MoE on an 8-16 GB consumer GPU), but the
    coordinator previously never forwarded the field, so the only way to set it
    was the untyped `llamaserver.extra_args` escape hatch.
    """
    from coordinator.models import ModelRegistry

    ModelRegistry.load_from_dict(
        {
            "models": {
                "moe-offload-test": {
                    "family": "qwen",
                    "parameters": "30B",
                    "min_memory_gb": 16,
                    "recommended_gpus": 1,
                    "max_gpus": 1,
                    "engine": "llamacpp",
                    "gguf": {
                        "repo_id": "org/repo",
                        "file": "model.gguf",
                        "n_cpu_moe": 40,
                    },
                }
            }
        }
    )
    model = ModelRegistry.get_model("moe-offload-test")
    assert model is not None
    assert model.gguf_n_cpu_moe == 40
    assert model.grpc_metadata()["n_cpu_moe"] == "40"


def test_gguf_n_cpu_moe_omitted_when_unset() -> None:
    """Absent `n_cpu_moe` must emit no metadata key (preserves prior behavior)."""
    from coordinator.models import ModelRegistry

    ModelRegistry.load_from_dict(
        {
            "models": {
                "moe-offload-unset": {
                    "family": "qwen",
                    "parameters": "30B",
                    "min_memory_gb": 16,
                    "recommended_gpus": 1,
                    "max_gpus": 1,
                    "engine": "llamacpp",
                    "gguf": {"repo_id": "org/repo", "file": "model.gguf"},
                }
            }
        }
    )
    model = ModelRegistry.get_model("moe-offload-unset")
    assert model is not None
    assert model.gguf_n_cpu_moe is None
    assert "n_cpu_moe" not in model.grpc_metadata()
