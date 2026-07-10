"""Coordinator-side context compression middleware.

Shrinks an incoming chat/completions request's prompt BEFORE it is forwarded
to a worker, when (and only when) it exceeds a configurable token budget. See
pending-work/12-context-compression-pipeline.md for the full design.

Public entrypoints (used by coordinator/api.py):
    maybe_compress_chat_messages() — for POST /v1/chat/completions
    maybe_compress_prompt()        — for POST /v1/completions
"""
from typing import TYPE_CHECKING, List, Optional, Sequence

from coordinator.context_compression.config import CompressionConfig
from coordinator.context_compression.pipeline import (
    InferProvider,
    Message,
    MessageLike,
    compress_messages,
    compress_prompt,
)

if TYPE_CHECKING:
    from coordinator.config import Settings


async def maybe_compress_chat_messages(
    messages: Sequence[MessageLike],
    *,
    coordinator: InferProvider,
    settings: Optional["Settings"] = None,
    override_enabled: Optional[bool] = None,
) -> List[Message]:
    """Entry point for `POST /v1/chat/completions` (coordinator/api.py).

    `messages` is anything sequence-like of objects with `.role`/`.content`
    (pydantic's `ChatMessage` satisfies this structurally — no conversion
    needed at the call site). `settings` defaults to `coordinator.settings`.
    """
    resolved_settings = settings if settings is not None else coordinator.settings  # type: ignore[attr-defined]
    config = CompressionConfig.from_settings(resolved_settings, override_enabled=override_enabled)
    outcome = await compress_messages(messages, config, infer_provider=coordinator)
    return outcome.messages


async def maybe_compress_prompt(
    prompt: str,
    *,
    coordinator: InferProvider,
    settings: Optional["Settings"] = None,
    override_enabled: Optional[bool] = None,
) -> str:
    """Entry point for `POST /v1/completions` (coordinator/api.py)."""
    resolved_settings = settings if settings is not None else coordinator.settings  # type: ignore[attr-defined]
    config = CompressionConfig.from_settings(resolved_settings, override_enabled=override_enabled)
    outcome = await compress_prompt(prompt, config, infer_provider=coordinator)
    return outcome.messages[-1].content if outcome.messages else prompt
