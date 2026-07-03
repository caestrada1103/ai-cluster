"""Compression pipeline orchestrator: budget check, no-op fast path, and the
technique dispatch loop (skeletonize here; summarize/llmlingua register their
handlers into `_TECHNIQUE_HANDLERS` in Tasks 9 and 10 without touching this
file's control flow).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Any, Awaitable, Callable, Dict, List, Optional, Protocol, Sequence, Tuple

from coordinator.context_compression.config import CompressionConfig
from coordinator.context_compression.segmenter import Segment, segment_text
from coordinator.context_compression.skeletonizer import skeletonize_segment
from coordinator.context_compression.tokenizer import apply_safety_margin, estimate_tokens

logger = logging.getLogger(__name__)

_PER_MESSAGE_OVERHEAD_TOKENS = 4  # rough chat-template scaffolding per turn


class InferProvider(Protocol):
    """Structural type for whatever can run the Phase-2 summarizer's
    sub-request — satisfied by `ClusterCoordinator` without any import of it
    here (avoids a coordinator.py <-> context_compression import cycle)."""

    async def infer(self, model_name: str, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        ...


class MessageLike(Protocol):
    """Structural type satisfied by both `Message` (frozen) and pydantic's
    `ChatMessage` (mutable) — read-only properties so a frozen dataclass's
    plain attributes still satisfy the protocol under mypy strict."""

    @property
    def role(self) -> str:
        ...

    @property
    def content(self) -> str:
        ...


@dataclass(frozen=True)
class Message:
    role: str
    content: str


@dataclass(frozen=True)
class CompressionOutcome:
    messages: List[Message]
    applied: Tuple[str, ...]
    tokens_before: int
    tokens_after: int
    skipped_reason: Optional[str]  # "disabled" | "under_budget" | None


TechniqueHandler = Callable[
    [List[Message], CompressionConfig, Optional[InferProvider]], Awaitable[List[Message]]
]

_TECHNIQUE_HANDLERS: Dict[str, TechniqueHandler] = {}


def _segment_all(messages: Sequence[Message]) -> List[List[Segment]]:
    return [segment_text(m.content) for m in messages]


def _estimate_total(messages: Sequence[Message]) -> int:
    total = 0
    for msg in messages:
        total += _PER_MESSAGE_OVERHEAD_TOKENS
        for seg in segment_text(msg.content):
            total += estimate_tokens(seg.text, is_code=(seg.kind == "code"))
    return total


def _under_budget(messages: Sequence[Message], config: CompressionConfig) -> bool:
    estimate = _estimate_total(messages)
    return apply_safety_margin(estimate, config.safety_margin) <= config.token_budget


async def _skeletonize_handler(
    messages: List[Message], config: CompressionConfig, _infer_provider: Optional[InferProvider]
) -> List[Message]:
    segmented = _segment_all(messages)

    # Global (across the whole conversation, in order) positions of every
    # code segment; the most recent `active_segments` of them are the
    # file/section being actively edited and are never touched.
    code_positions: List[Tuple[int, int]] = [
        (mi, si)
        for mi, segs in enumerate(segmented)
        for si, seg in enumerate(segs)
        if seg.kind == "code"
    ]
    protected = (
        set(code_positions[-config.active_segments :]) if config.active_segments > 0 else set()
    )

    new_messages: List[Message] = []
    for mi, (msg, segs) in enumerate(zip(messages, segmented, strict=True)):
        new_segs = list(segs)
        changed = False
        for si, seg in enumerate(segs):
            if seg.kind != "code" or (mi, si) in protected:
                continue
            new_seg, seg_changed = skeletonize_segment(seg)
            if seg_changed:
                new_segs[si] = new_seg
                changed = True
        new_messages.append(
            replace(msg, content="".join(s.text for s in new_segs)) if changed else msg
        )
    return new_messages


_TECHNIQUE_HANDLERS["skeletonize"] = _skeletonize_handler


async def compress_messages(
    messages: Sequence["MessageLike"],
    config: CompressionConfig,
    *,
    infer_provider: Optional[InferProvider] = None,
) -> CompressionOutcome:
    """Apply enabled techniques, in `config.techniques` order, until the
    estimated tokens (with safety margin) are back under budget or every
    technique has run. No-ops (fast, no segmentation work at all beyond one
    estimate pass) when disabled or already under budget."""
    working: List[Message] = [Message(role=m.role, content=m.content) for m in messages]

    if not config.enabled:
        return CompressionOutcome(
            messages=working, applied=(), tokens_before=0, tokens_after=0, skipped_reason="disabled"
        )

    tokens_before = _estimate_total(working)
    if apply_safety_margin(tokens_before, config.safety_margin) <= config.token_budget:
        return CompressionOutcome(
            messages=working,
            applied=(),
            tokens_before=tokens_before,
            tokens_after=tokens_before,
            skipped_reason="under_budget",
        )

    applied: List[str] = []
    for technique in config.techniques:
        handler = _TECHNIQUE_HANDLERS.get(technique)
        if handler is None:
            logger.debug(
                "context_compression: technique %r not available in this build; skipping",
                technique,
            )
            continue
        working = await handler(working, config, infer_provider)
        applied.append(technique)
        if _under_budget(working, config):
            break

    tokens_after = _estimate_total(working)
    logger.info(
        "context_compression: applied=%s tokens_before=%d tokens_after=%d budget=%d",
        applied,
        tokens_before,
        tokens_after,
        config.token_budget,
    )
    return CompressionOutcome(
        messages=working,
        applied=tuple(applied),
        tokens_before=tokens_before,
        tokens_after=tokens_after,
        skipped_reason=None,
    )


async def compress_prompt(
    prompt: str, config: CompressionConfig, *, infer_provider: Optional[InferProvider] = None
) -> CompressionOutcome:
    """Convenience wrapper for the raw-string `/v1/completions` endpoint —
    treats the prompt as a single synthetic user turn through the same
    pipeline `/v1/chat/completions` uses."""
    return await compress_messages(
        [Message(role="user", content=prompt)], config, infer_provider=infer_provider
    )
