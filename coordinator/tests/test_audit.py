"""Tests for coordinator.audit — record() emission, shape, and failure containment."""

import json
import logging
from datetime import datetime

import pytest

from coordinator.audit import (
    ACTION_MODEL_LOAD,
    OUTCOME_SUCCESS,
    record,
)

# ---------------------------------------------------------------------------
# Basic emission
# ---------------------------------------------------------------------------


def test_record_emits_exactly_one_line(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="coordinator.audit"):
        record(ACTION_MODEL_LOAD, caller="key-a", outcome=OUTCOME_SUCCESS)
    records = [r for r in caplog.records if r.name == "coordinator.audit"]
    assert len(records) == 1
    assert records[0].levelno == logging.INFO


def test_record_message_is_parseable_json(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="coordinator.audit"):
        record(ACTION_MODEL_LOAD, caller="key-a", outcome=OUTCOME_SUCCESS)
    payload = json.loads(caplog.records[0].message)
    assert payload["action"] == ACTION_MODEL_LOAD
    assert payload["caller"] == "key-a"
    assert payload["outcome"] == OUTCOME_SUCCESS


# ---------------------------------------------------------------------------
# Optional fields
# ---------------------------------------------------------------------------


def test_optional_fields_omitted_when_none(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="coordinator.audit"):
        record(ACTION_MODEL_LOAD, caller="key-a", outcome=OUTCOME_SUCCESS)
    payload = json.loads(caplog.records[0].message)
    assert "model" not in payload
    assert "worker" not in payload
    assert "detail" not in payload


def test_optional_fields_present_when_given(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="coordinator.audit"):
        record(
            ACTION_MODEL_LOAD,
            caller="key-a",
            outcome=OUTCOME_SUCCESS,
            model="llama-3-8b",
            worker="worker-1",
            detail="loaded fine",
        )
    payload = json.loads(caplog.records[0].message)
    assert payload["model"] == "llama-3-8b"
    assert payload["worker"] == "worker-1"
    assert payload["detail"] == "loaded fine"


def test_key_order_is_stable(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="coordinator.audit"):
        record(
            ACTION_MODEL_LOAD,
            caller="key-a",
            outcome=OUTCOME_SUCCESS,
            model="m",
            worker="w",
            detail="d",
        )
    assert list(json.loads(caplog.records[0].message).keys()) == [
        "ts",
        "action",
        "caller",
        "outcome",
        "model",
        "worker",
        "detail",
    ]


# ---------------------------------------------------------------------------
# Timestamp
# ---------------------------------------------------------------------------


def test_ts_parses_as_aware_utc_datetime(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="coordinator.audit"):
        record(ACTION_MODEL_LOAD, caller="key-a", outcome=OUTCOME_SUCCESS)
    payload = json.loads(caplog.records[0].message)
    parsed = datetime.fromisoformat(payload["ts"])
    assert parsed.tzinfo is not None
    offset = parsed.utcoffset()
    assert offset is not None
    assert offset.total_seconds() == 0


# ---------------------------------------------------------------------------
# Hostile / oversized input
# ---------------------------------------------------------------------------


def test_embedded_newlines_do_not_break_single_line(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="coordinator.audit"):
        record(
            ACTION_MODEL_LOAD,
            caller="key-a",
            outcome=OUTCOME_SUCCESS,
            detail="line one\nline two\r\nline three",
        )
    message = caplog.records[0].message
    assert "\n" not in message
    assert "\r" not in message
    payload = json.loads(message)
    assert "\n" in payload["detail"]


def test_overlong_detail_is_truncated(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="coordinator.audit"):
        record(ACTION_MODEL_LOAD, caller="key-a", outcome=OUTCOME_SUCCESS, detail="x" * 500)
    payload = json.loads(caplog.records[0].message)
    assert len(payload["detail"]) <= 200
    assert payload["detail"].endswith("...")


def test_overlong_caller_is_truncated(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="coordinator.audit"):
        record(ACTION_MODEL_LOAD, caller="k" * 500, outcome=OUTCOME_SUCCESS)
    payload = json.loads(caplog.records[0].message)
    assert len(payload["caller"]) <= 128
    assert payload["caller"].endswith("...")


# ---------------------------------------------------------------------------
# Total function — never raises
# ---------------------------------------------------------------------------


def test_record_swallows_internal_errors(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def _boom(*_args: object, **_kwargs: object) -> str:
        raise RuntimeError("boom")

    monkeypatch.setattr(json, "dumps", _boom)
    with caplog.at_level(logging.WARNING, logger="coordinator.audit"):
        record(ACTION_MODEL_LOAD, caller="key-a", outcome=OUTCOME_SUCCESS)
