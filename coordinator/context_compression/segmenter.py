"""Split message content into ordered code / natural-language segments.

Classification is purely syntactic (fenced-code-block detection) — no
language model, no heuristics beyond CommonMark-style fence matching. This is
the safety gate every downstream technique keys off: only `kind == "nl"`
segments are ever candidates for summarization/LLMLingua; `kind == "code"`
segments are only ever touched by the skeletonizer (Task 5), and only when
its own, stricter safety checks pass.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

# Matches an opening fence line: optional leading indent (<=3 spaces, per
# CommonMark), 3+ backticks or tildes, then an optional info string.
_FENCE_OPEN_RE = re.compile(r"^([ \t]{0,3})(`{3,}|~{3,})[ \t]*([^\n`~]*)?[ \t]*$")


@dataclass(frozen=True)
class Segment:
    kind: str  # "code" | "nl"
    text: str  # verbatim original text; segments concatenate back to the input exactly
    language: Optional[
        str
    ] = None  # only for kind == "code": lowercased first token of the info string
    inner: Optional[str] = None  # only for kind == "code": text strictly between the fence lines
    fence_open: Optional[
        str
    ] = None  # only for kind == "code": the opening fence line, incl. trailing \n
    fence_close: Optional[
        str
    ] = None  # only for kind == "code": closing fence line, or "" if unterminated


def _close_re_for(fence_char: str, fence_len: int) -> "re.Pattern[str]":
    return re.compile(rf"^[ \t]{{0,3}}{re.escape(fence_char)}{{{fence_len},}}[ \t]*$")


def segment_text(text: str) -> List[Segment]:
    """Split `text` into ordered Segments. Guarantees
    `"".join(s.text for s in segment_text(text)) == text` for all input."""
    lines = text.splitlines(keepends=True)
    segments: List[Segment] = []
    nl_buffer: List[str] = []
    i = 0

    def flush_nl() -> None:
        if nl_buffer:
            segments.append(Segment(kind="nl", text="".join(nl_buffer)))
            nl_buffer.clear()

    while i < len(lines):
        stripped = lines[i].rstrip("\n")
        match = _FENCE_OPEN_RE.match(stripped)
        if match is None:
            nl_buffer.append(lines[i])
            i += 1
            continue

        fence_marker = match.group(2)
        info_string = (match.group(3) or "").strip()
        fence_char, fence_len = fence_marker[0], len(fence_marker)
        close_re = _close_re_for(fence_char, fence_len)

        fence_open = lines[i]
        j = i + 1
        inner_lines: List[str] = []
        fence_close = ""
        while j < len(lines):
            if close_re.match(lines[j].rstrip("\n")):
                fence_close = lines[j]
                j += 1
                break
            inner_lines.append(lines[j])
            j += 1
        else:
            # Ran off the end without a closing fence: keep everything
            # gathered as code (never risk summarizing/truncating an
            # incomplete block — see test_unterminated_fence_is_kept_as_code).
            pass

        flush_nl()
        language = info_string.split()[0].lower() if info_string else None
        inner = "".join(inner_lines)
        full_text = fence_open + inner + fence_close
        segments.append(
            Segment(
                kind="code",
                text=full_text,
                language=language,
                inner=inner,
                fence_open=fence_open,
                fence_close=fence_close,
            )
        )
        i = j

    flush_nl()
    return segments
