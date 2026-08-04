"""Append-only audit log for management actions. Never logs prompts, bodies, or key material."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Optional

__all__ = [
    "record",
    "ACTION_MODEL_LOAD",
    "ACTION_MODEL_UNLOAD",
    "ACTION_MODEL_AUTOLOAD",
    "ACTION_MODEL_EVICTED",
    "ACTION_WORKER_REGISTER",
    "OUTCOME_SUCCESS",
    "OUTCOME_FAILURE",
    "OUTCOME_DENIED",
]

logger = logging.getLogger("coordinator.audit")

ACTION_MODEL_LOAD = "model.load"
ACTION_MODEL_UNLOAD = "model.unload"
ACTION_MODEL_AUTOLOAD = "model.autoload"
ACTION_MODEL_EVICTED = "model.evicted"
ACTION_WORKER_REGISTER = "worker.register"

OUTCOME_SUCCESS = "success"
OUTCOME_FAILURE = "failure"
OUTCOME_DENIED = "denied"

_DETAIL_MAX = 200
_FIELD_MAX = 128
_TRUNCATION_SUFFIX = "..."


def _truncate(value: str, limit: int) -> str:
    """Cap string length, appending a suffix when truncated."""
    if len(value) <= limit:
        return value
    return value[: max(limit - len(_TRUNCATION_SUFFIX), 0)] + _TRUNCATION_SUFFIX


def record(
    action: str,
    *,
    caller: str,
    outcome: str,
    model: Optional[str] = None,
    worker: Optional[str] = None,
    detail: Optional[str] = None,
) -> None:
    """Emit one single-line JSON audit record at INFO. Never raises."""
    try:
        entry: dict[str, str] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "caller": _truncate(caller, _FIELD_MAX),
            "outcome": outcome,
        }
        if model is not None:
            entry["model"] = _truncate(model, _FIELD_MAX)
        if worker is not None:
            entry["worker"] = _truncate(worker, _FIELD_MAX)
        if detail is not None:
            entry["detail"] = _truncate(detail, _DETAIL_MAX)
        line = json.dumps(entry, separators=(",", ":"))
        logger.info(line)
    except Exception as exc:  # noqa: BLE001 - audit must never break the caller
        try:
            logger.warning("audit record failed: %s", exc)
        except Exception:  # noqa: BLE001 - truly total, even logging can't raise out
            pass
