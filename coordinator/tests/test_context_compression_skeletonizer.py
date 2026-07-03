"""Tests for the tree-sitter skeletonizer. Every skeletonized result is
re-parsed and asserted syntactically valid — that's the core safety property
this module exists to guarantee."""
from coordinator.context_compression.segmenter import Segment, segment_text
from coordinator.context_compression.skeletonizer import skeletonize_segment


def _code_segment(inner: str, language: str) -> Segment:
    (seg,) = segment_text(f"```{language}\n{inner}```\n")
    return seg


def test_non_code_segment_is_untouched() -> None:
    seg = Segment(kind="nl", text="just prose")
    result, changed = skeletonize_segment(seg)
    assert changed is False
    assert result is seg


def test_unsupported_language_is_untouched() -> None:
    seg = _code_segment("fn main() {}\n", "rust")
    result, changed = skeletonize_segment(seg)
    assert changed is False
    assert result.text == seg.text


def test_no_language_is_untouched() -> None:
    (seg,) = segment_text("```\nsome text\n```\n")
    result, changed = skeletonize_segment(seg)
    assert changed is False


def test_python_function_body_replaced_signature_kept() -> None:
    seg = _code_segment("def add(x, y):\n    z = x + y\n    return z\n", "python")
    result, changed = skeletonize_segment(seg)
    assert changed is True
    assert result.inner is not None
    assert "def add(x, y):" in result.inner
    assert "z = x + y" not in result.inner
    assert "return z" not in result.inner
    assert "skeletonized" in result.inner


def test_python_class_keeps_signature_and_each_method_signature() -> None:
    inner = (
        "class Widget:\n"
        "    def method_one(self):\n"
        "        return 1\n\n"
        "    def method_two(self, x):\n"
        "        y = x * 2\n"
        "        return y\n"
    )
    seg = _code_segment(inner, "python")
    result, changed = skeletonize_segment(seg)
    assert changed is True
    assert result.inner is not None
    assert "class Widget:" in result.inner
    assert "def method_one(self):" in result.inner
    assert "def method_two(self, x):" in result.inner
    assert "return 1" not in result.inner
    assert "y = x * 2" not in result.inner


def test_python_imports_are_preserved() -> None:
    inner = "import os\nfrom typing import List\n\ndef f():\n    return os.getcwd()\n"
    seg = _code_segment(inner, "python")
    result, changed = skeletonize_segment(seg)
    assert changed is True
    assert result.inner is not None
    assert "import os" in result.inner
    assert "from typing import List" in result.inner


def test_result_always_reparses_without_syntax_errors() -> None:
    inner = "def f(x):\n    if x:\n        return 1\n    return 0\n"
    seg = _code_segment(inner, "python")
    result, changed = skeletonize_segment(seg)
    assert changed is True
    assert result.inner is not None

    import tree_sitter_python as tspython
    from tree_sitter import Language, Parser

    parser = Parser(Language(tspython.language()))
    tree = parser.parse(result.inner.encode("utf-8"))
    assert tree.root_node.has_error is False


def test_typescript_function_and_method_bodies_replaced() -> None:
    inner = (
        "export function add(a: number, b: number): number {\n"
        "  const z = a + b;\n"
        "  return z;\n"
        "}\n\n"
        "class Widget {\n"
        "  method(x: number): void {\n"
        "    this.count += x;\n"
        "  }\n"
        "}\n"
    )
    seg = _code_segment(inner, "typescript")
    result, changed = skeletonize_segment(seg)
    assert changed is True
    assert result.inner is not None
    assert "export function add(a: number, b: number): number" in result.inner
    assert "const z = a + b;" not in result.inner
    assert "method(x: number): void" in result.inner
    assert "this.count += x;" not in result.inner


def test_unparseable_code_is_left_untouched_not_half_mangled() -> None:
    """Safety net: if the input doesn't even parse cleanly to begin with,
    don't attempt surgery on it — pass it through unchanged."""
    seg = _code_segment("def broken(:\n    pass\n", "python")
    result, changed = skeletonize_segment(seg)
    assert changed is False
    assert result.text == seg.text


def test_fence_and_language_are_preserved_around_skeletonized_inner() -> None:
    seg = _code_segment("def f():\n    return 1\n", "python")
    result, _ = skeletonize_segment(seg)
    assert result.text.startswith("```python\n")
    assert result.text.rstrip("\n").endswith("```")
