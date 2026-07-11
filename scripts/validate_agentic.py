#!/usr/bin/env python3
"""Validate a live coordinator's agentic-serving surface (Plan 13 Phase 2 Task 8).

Run this against a real, running coordinator (+ at least one worker with a
``engine = "llamaserver"`` model loaded, see ``config/models.toml``) to sanity
check the OpenAI- and Anthropic-compatible endpoints end to end: model
listing, chat completions (streaming + non-streaming), tool calling, the
Anthropic ``/v1/messages`` surface, and opt-in API-key auth.

Deliberately **stdlib-only** (``urllib``/``json``/``argparse``, no
``httpx``/``requests``) so the owner can run it on any machine with Python
3.10+ and no ``pip install`` step — see ``scripts/requirements-scripts.txt``
for the (heavier) convention other scripts in this directory follow; this one
does not need it.

Usage:
    python scripts/validate_agentic.py
    python scripts/validate_agentic.py --base-url http://192.168.1.50:8000
    python scripts/validate_agentic.py --model devstral-small-2-24b-instruct-gguf
    python scripts/validate_agentic.py --api-key sk-my-secret   # also exercises auth checks

Exit code = number of FAILed checks (0 means everything passed; SKIPs do not
count against the exit code).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Iterator, List, Optional, Tuple

DEFAULT_BASE_URL = "http://localhost:8000"
# Generous default: a CPU-offloaded MoE model (e.g. qwen3-coder-30b-a3b with
# --n-cpu-moe) can take a while to first token on real hardware. This is a
# per-socket-operation timeout (each read must complete within this window),
# not a hard cap on total generation time, so a large value is safe even on
# fast hardware.
DEFAULT_TIMEOUT = 120.0
# Small generation budgets keep this script fast to run repeatedly while
# still exercising streaming/tool-call code paths meaningfully.
CHAT_MAX_TOKENS = 64
TOOLS_MAX_TOKENS = 256

# Keep in sync with config/models.toml's `engine = "llamaserver"` entries.
# GET /v1/models does not currently expose each model's engine (see
# coordinator/api.py::ModelInfo / coordinator/coordinator.py::list_models),
# so this script cannot ask the API "which model is llamaserver" directly.
# Used only as a *default-selection* fallback when --model is omitted: an
# "engine" field in the API response is honored first (see check_models),
# in case a future coordinator change starts exposing it.
KNOWN_LLAMASERVER_MODELS: Tuple[str, ...] = (
    "devstral-small-2-24b-instruct-gguf",
    "qwen3-coder-30b-a3b-instruct-gguf",
)

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"

GET_WEATHER_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name, e.g. 'Paris'",
                }
            },
            "required": ["city"],
        },
    },
}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


class Report:
    """Accumulates PASS/FAIL/SKIP lines and prints each one as it happens."""

    def __init__(self) -> None:
        self.counts: Dict[str, int] = {PASS: 0, FAIL: 0, SKIP: 0}

    def log(self, status: str, name: str, detail: str = "") -> None:
        self.counts[status] = self.counts.get(status, 0) + 1
        line = f"[{status:4}] {name}"
        if detail:
            line += f" - {detail}"
        print(line)

    @property
    def fail_count(self) -> int:
        return self.counts[FAIL]

    def summary(self) -> str:
        return (
            f"Summary: {self.counts[PASS]} passed, {self.counts[FAIL]} failed, "
            f"{self.counts[SKIP]} skipped"
        )


# ---------------------------------------------------------------------------
# Minimal stdlib HTTP helpers
# ---------------------------------------------------------------------------


class ConnectionProblem(Exception):
    """Network-level failure (refused, DNS, timeout) — never a coordinator response."""


class HttpResult:
    def __init__(self, status: int, headers: Dict[str, str], body: bytes) -> None:
        self.status = status
        self.headers = headers
        self.body = body


def _auth_headers(api_key: Optional[str]) -> Dict[str, str]:
    if not api_key:
        return {}
    return {"Authorization": f"Bearer {api_key}"}


def http_call(
    url: str,
    method: str = "GET",
    json_body: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> HttpResult:
    """A single non-streaming HTTP round trip. Raises ConnectionProblem on network errors."""
    data = None
    req_headers = dict(headers or {})
    if json_body is not None:
        data = json.dumps(json_body).encode("utf-8")
        req_headers.setdefault("Content-Type", "application/json")
    request = urllib.request.Request(url, data=data, method=method, headers=req_headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            return HttpResult(resp.status, dict(resp.headers.items()), resp.read())
    except urllib.error.HTTPError as exc:
        body = exc.read()
        resp_headers = dict(exc.headers.items()) if exc.headers else {}
        return HttpResult(exc.code, resp_headers, body)
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
        raise ConnectionProblem(str(exc)) from exc


def open_stream(
    url: str,
    json_body: Dict[str, Any],
    headers: Optional[Dict[str, str]],
    timeout: float,
) -> Any:
    """Open a streaming POST and return the live (unread) response object.

    Both a normal 2xx response and an ``HTTPError`` are file-like objects
    supporting ``.read()`` / line iteration / ``.close()``, so callers can
    treat the return value uniformly and just check ``_resp_status()``.
    """
    data = json.dumps(json_body).encode("utf-8")
    req_headers = dict(headers or {})
    req_headers.setdefault("Content-Type", "application/json")
    request = urllib.request.Request(url, data=data, method="POST", headers=req_headers)
    try:
        return urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        return exc
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
        raise ConnectionProblem(str(exc)) from exc


def _resp_status(resp: Any) -> Optional[int]:
    return getattr(resp, "status", None) or getattr(resp, "code", None)


def iter_sse_events(resp: Any) -> Iterator[Dict[str, Optional[str]]]:
    """Yield ``{"event": <event-name-or-None>, "data": <joined-data-lines>}`` dicts.

    Handles both the OpenAI style (``data:`` lines only, ``[DONE]`` sentinel)
    and the Anthropic style (explicit ``event:`` lines before ``data:``).
    Comment lines (``:...``) are ignored per the SSE spec.
    """
    event_type: Optional[str] = None
    data_lines: List[str] = []
    for raw_line in resp:
        line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
        if line == "":
            if data_lines:
                yield {"event": event_type, "data": "\n".join(data_lines)}
            event_type = None
            data_lines = []
            continue
        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            event_type = line[len("event:") :].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:") :].strip())
    if data_lines:
        yield {"event": event_type, "data": "\n".join(data_lines)}


def _short(data: bytes, limit: int = 220) -> str:
    text = data.decode("utf-8", errors="replace").strip()
    if len(text) > limit:
        text = text[:limit] + "..."
    return text or "<empty body>"


def _connection_hint(exc: ConnectionProblem) -> str:
    return (
        f"could not connect ({exc}) - is the coordinator running at this --base-url? "
        "(e.g. `docker compose up -d coordinator` or "
        "`uvicorn coordinator.main:app --host 0.0.0.0 --port 8000`); "
        "double-check the host/port and any firewall rules"
    )


def _status_hint(status: Optional[int]) -> str:
    if status == 401:
        return (
            " (coordinator requires an API key - COORDINATOR_API_KEYS is set on the "
            "server; retry with --api-key, or check the key is correct)"
        )
    if status == 404:
        return (
            " (model may not be loaded on any worker - try "
            '`POST /v1/models/load {"model_name": "<model>"}` first)'
        )
    if status == 501:
        return (
            " (this model's engine does not support this endpoint - only "
            'engine="llamaserver" models serve /v1/messages)'
        )
    if status is not None and status >= 500:
        return " (coordinator/worker-side error - check coordinator and worker logs)"
    return ""


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def check_health(base_url: str, headers: Dict[str, str], timeout: float, report: Report) -> None:
    name = "a. GET /health"
    try:
        r = http_call(f"{base_url}/health", headers=headers, timeout=timeout)
    except ConnectionProblem as exc:
        report.log(FAIL, name, _connection_hint(exc))
        return
    if r.status != 200:
        report.log(FAIL, name, f"HTTP {r.status}: {_short(r.body)}{_status_hint(r.status)}")
        return
    try:
        data = json.loads(r.body)
    except json.JSONDecodeError as exc:
        report.log(FAIL, name, f"invalid JSON: {exc}")
        return
    report.log(PASS, name, f"status={data.get('status')!r} workers={data.get('workers')!r}")


def check_models(
    base_url: str,
    headers: Dict[str, str],
    requested_model: Optional[str],
    timeout: float,
    report: Report,
) -> Optional[str]:
    """b. GET /v1/models lists the target model.

    Also resolves the auto-picked default model when --model was not given.
    Returns the target model name, or None if it could not be resolved.
    """
    name = "b. GET /v1/models (target model listed)"
    try:
        r = http_call(f"{base_url}/v1/models", headers=headers, timeout=timeout)
    except ConnectionProblem as exc:
        report.log(FAIL, name, _connection_hint(exc))
        return None
    if r.status != 200:
        report.log(FAIL, name, f"HTTP {r.status}: {_short(r.body)}{_status_hint(r.status)}")
        return None
    try:
        data = json.loads(r.body)
    except json.JSONDecodeError as exc:
        report.log(FAIL, name, f"invalid JSON: {exc}")
        return None
    entries = data.get("data")
    if not isinstance(entries, list):
        report.log(FAIL, name, "response has no 'data' list")
        return None
    ids = [e.get("id") for e in entries if isinstance(e, dict) and e.get("id")]

    if requested_model:
        if requested_model in ids:
            report.log(PASS, name, f"'{requested_model}' found among {len(ids)} model(s)")
            return requested_model
        report.log(FAIL, name, f"'{requested_model}' NOT found; available: {ids or '(none)'}")
        return None

    # No --model given: prefer an explicit "engine" field if the API response
    # ever carries one (forward-compat), else fall back to the known
    # llamaserver-engine names from config/models.toml.
    for entry in entries:
        if isinstance(entry, dict) and entry.get("engine") == "llamaserver" and entry.get("id"):
            picked = str(entry["id"])
            report.log(PASS, name, f"auto-picked '{picked}' (engine=llamaserver in response)")
            return picked
    for candidate in KNOWN_LLAMASERVER_MODELS:
        if candidate in ids:
            report.log(PASS, name, f"auto-picked '{candidate}' (known llamaserver model)")
            return candidate

    report.log(
        FAIL,
        name,
        "no --model given and none of the known llamaserver models "
        f"({', '.join(KNOWN_LLAMASERVER_MODELS)}) are registered on this coordinator; "
        f"pass --model explicitly. available models: {ids or '(none)'}",
    )
    return None


def check_nonstream_chat(
    base_url: str, headers: Dict[str, str], model: str, timeout: float, report: Report
) -> None:
    name = "c. POST /v1/chat/completions (non-stream)"
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with the single word: pong"}],
        "max_tokens": CHAT_MAX_TOKENS,
        "stream": False,
    }
    try:
        r = http_call(
            f"{base_url}/v1/chat/completions",
            method="POST",
            json_body=body,
            headers=headers,
            timeout=timeout,
        )
    except ConnectionProblem as exc:
        report.log(FAIL, name, _connection_hint(exc))
        return
    if r.status != 200:
        report.log(FAIL, name, f"HTTP {r.status}: {_short(r.body)}{_status_hint(r.status)}")
        return
    try:
        data = json.loads(r.body)
    except json.JSONDecodeError as exc:
        report.log(FAIL, name, f"invalid JSON: {exc}")
        return
    choices = data.get("choices") or []
    content = (choices[0].get("message") or {}).get("content") if choices else None
    usage = data.get("usage")
    usage_present = isinstance(usage, dict)
    prompt_tokens = usage.get("prompt_tokens") if usage_present else None
    prompt_tokens_positive = isinstance(prompt_tokens, (int, float)) and prompt_tokens > 0
    ok = bool(content) and usage_present
    detail = (
        f"content={'present' if content else 'MISSING'}, "
        f"usage={'present' if usage_present else 'MISSING'}, "
        f"usage.prompt_tokens={prompt_tokens!r} "
        f"({'>0' if prompt_tokens_positive else 'not >0 - informational only'})"
    )
    report.log(PASS if ok else FAIL, name, detail)


def check_stream_chat(
    base_url: str, headers: Dict[str, str], model: str, timeout: float, report: Report
) -> Dict[str, Any]:
    """d. POST /v1/chat/completions with stream: true.

    Returns a stats dict (chunks/elapsed/text/ok) used by the timing check (h),
    regardless of whether this check itself passed.
    """
    name = "d. POST /v1/chat/completions (stream)"
    stats: Dict[str, Any] = {"chunks": 0, "elapsed": 0.0, "text": "", "ok": False}
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "Count from 1 to 5, one number per line."}],
        "max_tokens": CHAT_MAX_TOKENS,
        "stream": True,
    }
    start = time.monotonic()
    try:
        resp = open_stream(f"{base_url}/v1/chat/completions", body, headers, timeout)
    except ConnectionProblem as exc:
        report.log(FAIL, name, _connection_hint(exc))
        return stats

    status = _resp_status(resp)
    if status != 200:
        raw = resp.read()
        resp.close()
        report.log(FAIL, name, f"HTTP {status}: {_short(raw)}{_status_hint(status)}")
        return stats

    chunks = 0
    text_parts: List[str] = []
    done_seen = False
    finish_reason: Optional[str] = None
    last_at = start
    try:
        for event in iter_sse_events(resp):
            data = event["data"]
            if data == "[DONE]":
                done_seen = True
                last_at = time.monotonic()
                break
            try:
                chunk = json.loads(data) if data else {}
            except json.JSONDecodeError:
                continue
            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            content = delta.get("content")
            if content:
                chunks += 1
                text_parts.append(content)
                last_at = time.monotonic()
            fr = choices[0].get("finish_reason")
            if fr:
                finish_reason = fr
    except (TimeoutError, OSError) as exc:
        report.log(FAIL, name, f"stream read error: {exc}")
        resp.close()
        return stats
    finally:
        resp.close()

    stats["chunks"] = chunks
    stats["elapsed"] = max(last_at - start, 0.0)
    stats["text"] = "".join(text_parts)
    ok = chunks > 0 and done_seen and bool(stats["text"].strip())
    stats["ok"] = ok
    report.log(
        PASS if ok else FAIL,
        name,
        f"{chunks} content chunk(s), done_seen={done_seen}, "
        f"finish_reason={finish_reason!r}, {stats['elapsed']:.2f}s",
    )
    return stats


def check_tools(
    base_url: str, headers: Dict[str, str], model: str, timeout: float, report: Report
) -> None:
    name = "e. tools round-trip (get_weather)"
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "What is the weather in Paris? Use the tool."}],
        "tools": [GET_WEATHER_TOOL],
        "max_tokens": TOOLS_MAX_TOKENS,
        "stream": False,
    }
    try:
        r = http_call(
            f"{base_url}/v1/chat/completions",
            method="POST",
            json_body=body,
            headers=headers,
            timeout=timeout,
        )
    except ConnectionProblem as exc:
        report.log(FAIL, name, _connection_hint(exc))
        return
    if r.status != 200:
        report.log(FAIL, name, f"HTTP {r.status}: {_short(r.body)}{_status_hint(r.status)}")
        return
    try:
        data = json.loads(r.body)
    except json.JSONDecodeError as exc:
        report.log(FAIL, name, f"invalid JSON: {exc}")
        return
    choices = data.get("choices") or []
    if not choices:
        report.log(FAIL, name, "no choices in response")
        return
    message = choices[0].get("message") or {}
    finish_reason = choices[0].get("finish_reason")
    tool_calls = message.get("tool_calls")
    if not tool_calls:
        report.log(
            FAIL,
            name,
            f"no tool_calls in response (finish_reason={finish_reason!r}) - the model may "
            "not support llama.cpp's tool-calling grammar for its chat template, or ignored "
            "the instruction; try a different model or check llama-server's --jinja/template",
        )
        return

    parse_errors: List[str] = []
    for call in tool_calls:
        fn = (call or {}).get("function") or {}
        args_raw = fn.get("arguments")
        if not isinstance(args_raw, str):
            parse_errors.append(f"arguments not a JSON string (got {type(args_raw).__name__})")
            continue
        try:
            json.loads(args_raw)
        except json.JSONDecodeError as exc:
            parse_errors.append(f"arguments not valid JSON: {exc}")

    finish_reason_note = ""
    if finish_reason == "tool":
        finish_reason_note = " (non-standard but accepted - see Plan 13 'Honest risks')"
    elif finish_reason != "tool_calls":
        finish_reason_note = " (unexpected finish_reason, informational only)"
    detail = f"{len(tool_calls)} tool_call(s), finish_reason={finish_reason!r}{finish_reason_note}"
    if parse_errors:
        detail += "; " + "; ".join(parse_errors)
    report.log(PASS if not parse_errors else FAIL, name, detail)


def check_messages_nonstream(
    base_url: str, headers: Dict[str, str], model: str, timeout: float, report: Report
) -> None:
    name = "f. POST /v1/messages (non-stream)"
    body = {
        "model": model,
        "max_tokens": CHAT_MAX_TOKENS,
        "messages": [{"role": "user", "content": "Reply with the single word: pong"}],
    }
    try:
        r = http_call(
            f"{base_url}/v1/messages",
            method="POST",
            json_body=body,
            headers=headers,
            timeout=timeout,
        )
    except ConnectionProblem as exc:
        report.log(FAIL, name, _connection_hint(exc))
        return
    if r.status != 200:
        report.log(FAIL, name, f"HTTP {r.status}: {_short(r.body)}{_status_hint(r.status)}")
        return
    try:
        data = json.loads(r.body)
    except json.JSONDecodeError as exc:
        report.log(FAIL, name, f"invalid JSON: {exc}")
        return
    content = data.get("content")
    ok = isinstance(content, list) and len(content) > 0
    detail = f"content block(s)={len(content) if ok else content!r}"
    report.log(PASS if ok else FAIL, name, detail)


def check_messages_stream(
    base_url: str, headers: Dict[str, str], model: str, timeout: float, report: Report
) -> None:
    name = "f. POST /v1/messages (stream)"
    body = {
        "model": model,
        "max_tokens": CHAT_MAX_TOKENS,
        "messages": [{"role": "user", "content": "Count from 1 to 5."}],
        "stream": True,
    }
    try:
        resp = open_stream(f"{base_url}/v1/messages", body, headers, timeout)
    except ConnectionProblem as exc:
        report.log(FAIL, name, _connection_hint(exc))
        return
    status = _resp_status(resp)
    if status != 200:
        raw = resp.read()
        resp.close()
        report.log(FAIL, name, f"HTTP {status}: {_short(raw)}{_status_hint(status)}")
        return

    seen_events: set = set()
    try:
        for event in iter_sse_events(resp):
            event_name = event["event"]
            if event_name:
                seen_events.add(event_name)
                continue
            # Some servers omit explicit `event:` lines and only set "type"
            # inside the JSON payload - fall back to that.
            data = event["data"]
            if not data or data == "[DONE]":
                continue
            try:
                payload = json.loads(data)
            except json.JSONDecodeError:
                continue
            payload_type = payload.get("type") if isinstance(payload, dict) else None
            if payload_type:
                seen_events.add(payload_type)
    finally:
        resp.close()

    required = {"message_start", "content_block_delta", "message_stop"}
    missing = required - seen_events
    ok = not missing
    detail = f"event types seen: {sorted(seen_events) or '(none)'}"
    if missing:
        detail += f"; missing: {sorted(missing)}"
    report.log(PASS if ok else FAIL, name, detail)


def check_auth(base_url: str, api_key: Optional[str], timeout: float, report: Report) -> None:
    group = "g. auth (COORDINATOR_API_KEYS)"
    if not api_key:
        report.log(SKIP, f"{group}: without key / wrong key / correct key", "no --api-key given")
        return

    url = f"{base_url}/v1/models"
    scenarios = [
        ("without key", {}, 401),
        ("wrong key", {"x-api-key": api_key + "-wrong"}, 401),
        ("correct key", _auth_headers(api_key), 200),
    ]
    for label, scenario_headers, expected in scenarios:
        name = f"{group}: {label}"
        try:
            r = http_call(url, headers=scenario_headers, timeout=timeout)
        except ConnectionProblem as exc:
            report.log(FAIL, name, _connection_hint(exc))
            continue
        ok = r.status == expected
        detail = f"HTTP {r.status} (expected {expected})"
        if not ok and expected == 401 and r.status == 200:
            detail += " - is COORDINATOR_API_KEYS actually set on the coordinator?"
        report.log(PASS if ok else FAIL, name, detail)


def check_timing(stream_stats: Optional[Dict[str, Any]], report: Report) -> None:
    name = "h. timing summary (approx tokens/sec, from check d)"
    if stream_stats is None:
        report.log(SKIP, name, "no target model resolved (see check b)")
        return
    chunks = stream_stats.get("chunks", 0)
    elapsed = stream_stats.get("elapsed", 0.0)
    if not chunks or elapsed <= 0:
        report.log(SKIP, name, "no usable streaming data captured (see check d)")
        return
    tps = chunks / elapsed
    report.log(
        PASS,
        name,
        f"~{tps:.1f} tok/s ({chunks} SSE content chunks / {elapsed:.2f}s; approximate - "
        "one SSE chunk is not guaranteed to be exactly one token)",
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a live AI Cluster coordinator's agentic-serving surface "
            "(Plan 13 Phase 2 Task 8): models list, chat completions (stream + "
            "non-stream), tool calling, Anthropic /v1/messages, and optional "
            "API-key auth. Stdlib-only - no pip install required."
        )
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"Coordinator base URL (default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Target model name. Default: auto-pick the first known "
            f"llamaserver-engine model from GET /v1/models (currently: "
            f"{', '.join(KNOWN_LLAMASERVER_MODELS)}); required if none of "
            "those are registered on this coordinator."
        ),
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help=(
            "API key for COORDINATOR_API_KEYS auth, sent as "
            "'Authorization: Bearer <key>'. Also enables the auth checks (g)."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help=(
            "Per-request timeout in seconds (default: %(default)s - generous, "
            "since CPU-offloaded MoE models can be slow on real hardware)"
        ),
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    base_url = args.base_url.rstrip("/")
    headers = _auth_headers(args.api_key)
    timeout = args.timeout
    report = Report()

    print(f"Validating agentic serving at {base_url}")
    if args.model:
        print(f"Target model: {args.model} (from --model)")
    print()

    check_health(base_url, headers, timeout, report)
    model = check_models(base_url, headers, args.model, timeout, report)

    stream_stats: Optional[Dict[str, Any]] = None
    if model is None:
        for skipped_name in (
            "c. POST /v1/chat/completions (non-stream)",
            "d. POST /v1/chat/completions (stream)",
            "e. tools round-trip (get_weather)",
            "f. POST /v1/messages (non-stream)",
            "f. POST /v1/messages (stream)",
        ):
            report.log(SKIP, skipped_name, "no target model resolved (see check b)")
    else:
        check_nonstream_chat(base_url, headers, model, timeout, report)
        stream_stats = check_stream_chat(base_url, headers, model, timeout, report)
        check_tools(base_url, headers, model, timeout, report)
        check_messages_nonstream(base_url, headers, model, timeout, report)
        check_messages_stream(base_url, headers, model, timeout, report)

    check_auth(base_url, args.api_key, timeout, report)
    check_timing(stream_stats, report)

    print()
    print(report.summary())
    return report.fail_count


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
