"""Frozen, plain-dataclass view of the context-compression settings.

Decouples the rest of `context_compression/` from `pydantic`/`Settings` so
every other module in this package can be unit-tested without constructing a
full `coordinator.config.Settings` (or a FastAPI app) at all.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:
    from coordinator.config import Settings


@dataclass(frozen=True)
class CompressionConfig:
    enabled: bool
    token_budget: int
    safety_margin: float
    active_segments: int
    techniques: Tuple[str, ...]
    summarizer_model: str
    summarizer_max_tokens: int
    llmlingua_model: str
    llmlingua_rate: float

    @classmethod
    def from_settings(
        cls, settings: "Settings", *, override_enabled: Optional[bool] = None
    ) -> "CompressionConfig":
        """Build a `CompressionConfig` from `Settings`.

        `override_enabled` is the per-request `compress_context` field
        (`ChatCompletionRequest`/`CompletionRequest`, api.py) — `None` means
        "use the server default" (`settings.context_compression_enabled`);
        `True`/`False` force the middleware on/off for this request only.
        """
        enabled = (
            settings.context_compression_enabled if override_enabled is None else override_enabled
        )
        return cls(
            enabled=enabled,
            token_budget=settings.context_compression_token_budget,
            safety_margin=settings.context_compression_safety_margin,
            active_segments=settings.context_compression_active_segments,
            techniques=tuple(settings.context_compression_techniques),
            summarizer_model=settings.context_compression_summarizer_model,
            summarizer_max_tokens=settings.context_compression_summarizer_max_tokens,
            llmlingua_model=settings.context_compression_llmlingua_model,
            llmlingua_rate=settings.context_compression_llmlingua_rate,
        )
