"""Tests for the code/NL segmenter. The round-trip invariant (Step 1's last
test) is the most important one in this file: skeletonize/summarize are only
safe to build on top of a segmenter that never drops a byte."""
from coordinator.context_compression.segmenter import segment_text


def test_plain_text_is_one_nl_segment() -> None:
    segs = segment_text("just some prose, no code here")
    assert len(segs) == 1
    assert segs[0].kind == "nl"


def test_single_fenced_code_block() -> None:
    text = "before\n```python\ndef f():\n    pass\n```\nafter\n"
    segs = segment_text(text)
    kinds = [s.kind for s in segs]
    assert kinds == ["nl", "code", "nl"]
    assert segs[1].language == "python"
    assert segs[1].inner == "def f():\n    pass\n"


def test_language_alias_is_lowercased_first_token() -> None:
    text = "```TS\nconst x = 1;\n```\n"
    segs = segment_text(text)
    assert segs[0].language == "ts"


def test_no_language_info_string() -> None:
    text = "```\nplain fenced text\n```\n"
    segs = segment_text(text)
    assert segs[0].kind == "code"
    assert segs[0].language is None


def test_tilde_fence_supported() -> None:
    text = "~~~python\nx = 1\n~~~\n"
    segs = segment_text(text)
    assert segs[0].kind == "code"
    assert segs[0].inner == "x = 1\n"


def test_multiple_fenced_blocks_preserve_order() -> None:
    text = "one\n```py\na=1\n```\ntwo\n```py\nb=2\n```\nthree\n"
    segs = segment_text(text)
    assert [s.kind for s in segs] == ["nl", "code", "nl", "code", "nl"]


def test_unterminated_fence_is_kept_as_code_not_lost() -> None:
    """An unterminated fence must never be silently dropped or treated as
    summarizable prose — treat the remainder as code (the safer default)."""
    text = "before\n```python\ndef f():\n    pass\n"
    segs = segment_text(text)
    assert segs[-1].kind == "code"
    assert segs[-1].fence_close == ""


def test_round_trip_never_loses_a_byte() -> None:
    samples = [
        "",
        "no code",
        "```py\nx=1\n```",
        "prefix\n```js\nconsole.log(1)\n```\nsuffix",
        "~~~\nfenced no lang\n~~~\ntrailing text",
        "```python\nunterminated\n",
    ]
    for text in samples:
        segs = segment_text(text)
        assert "".join(s.text for s in segs) == text, f"round-trip failed for {text!r}"
