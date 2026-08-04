"""Deterministic, tree-sitter-backed code skeletonization.

Replaces function/method BODIES with a short placeholder comment, keeping
every signature, class header, and import fully intact. The result is always
re-parsed and checked for zero syntax errors before it is accepted — on any
failure (unsupported language, parse error, or a skeletonized result that
somehow doesn't re-parse cleanly) the original segment is returned untouched.
Code is never partially or speculatively rewritten.
"""
from __future__ import annotations

import logging
from dataclasses import replace
from functools import lru_cache
from typing import TYPE_CHECKING, Dict, FrozenSet, List, Tuple

from coordinator.context_compression.segmenter import Segment

if TYPE_CHECKING:
    from tree_sitter import Language, Node

logger = logging.getLogger(__name__)

# Fence info-string aliases -> canonical grammar key. Deliberately small —
# see pending-work/12-context-compression-pipeline.md's "Honest risks":
# extending this is the natural follow-up, not attempted here.
_LANGUAGE_ALIASES: Dict[str, str] = {
    "python": "python",
    "py": "python",
    "python3": "python",
    "typescript": "typescript",
    "ts": "typescript",
    "tsx": "tsx",
}

# Node types whose `body` field gets replaced wholesale. Anything NOT in this
# set (e.g. class_definition/class_declaration) is walked into, not cut, so
# nested declarations (methods) are still found and skeletonized individually.
_CUT_NODE_TYPES: Dict[str, FrozenSet[str]] = {
    "python": frozenset({"function_definition"}),
    "typescript": frozenset(
        {
            "function_declaration",
            "method_definition",
            "function_expression",
            "generator_function_declaration",
        }
    ),
    "tsx": frozenset(
        {
            "function_declaration",
            "method_definition",
            "function_expression",
            "generator_function_declaration",
        }
    ),
}


@lru_cache(maxsize=None)
def _load_language(canonical: str) -> "Language":
    from tree_sitter import Language

    if canonical == "python":
        import tree_sitter_python as ts_python

        return Language(ts_python.language())
    if canonical == "typescript":
        import tree_sitter_typescript as ts_typescript

        return Language(ts_typescript.language_typescript())
    if canonical == "tsx":
        import tree_sitter_typescript as ts_tsx

        return Language(ts_tsx.language_tsx())
    raise KeyError(canonical)


def _placeholder(canonical: str) -> bytes:
    # Deliberately terse (must stay shorter than the bodies it replaces) and
    # never executed — only re-parsed for syntax validity.
    if canonical == "python":
        return b"skeletonized"
    return b"{ /* skeletonized */ }"


def _collect_cuts(node: "Node", cut_types: FrozenSet[str], cuts: List["Node"]) -> None:
    """Walk the tree, recording body nodes to replace. Recurses into every
    child EXCEPT a body we just decided to cut (nothing worth exploring
    inside a range about to become one placeholder line)."""
    body = node.child_by_field_name("body") if node.type in cut_types else None
    for child in node.named_children:
        if body is not None and child.id == body.id:
            cuts.append(child)
            continue
        _collect_cuts(child, cut_types, cuts)


def _skeletonize_source(source: str, canonical: str) -> Tuple[str, bool]:
    from tree_sitter import Parser

    language = _load_language(canonical)
    parser = Parser(language)
    source_bytes = source.encode("utf-8")

    tree = parser.parse(source_bytes)
    if tree.root_node.has_error:
        return source, False  # don't operate on already-broken input

    cuts: List["Node"] = []
    _collect_cuts(tree.root_node, _CUT_NODE_TYPES[canonical], cuts)
    if not cuts:
        return source, False

    placeholder = _placeholder(canonical)
    for body in sorted(cuts, key=lambda n: n.start_byte, reverse=True):
        source_bytes = source_bytes[: body.start_byte] + placeholder + source_bytes[body.end_byte :]

    new_source = source_bytes.decode("utf-8")

    # Safety net: the result MUST re-parse with zero syntax errors, or we
    # discard it and keep the original — never ship a half-mangled result.
    verify_tree = parser.parse(new_source.encode("utf-8"))
    if verify_tree.root_node.has_error:
        logger.warning(
            "context_compression: skeletonized %s source failed to re-parse cleanly; "
            "keeping original",
            canonical,
        )
        return source, False

    return new_source, True


def skeletonize_segment(segment: Segment) -> Tuple[Segment, bool]:
    """Skeletonize one code Segment. Returns `(segment, False)` unchanged for
    non-code segments, unsupported/absent languages, or any failure."""
    if segment.kind != "code" or not segment.language or segment.inner is None:
        return segment, False

    canonical = _LANGUAGE_ALIASES.get(segment.language)
    if canonical is None:
        return segment, False

    try:
        new_inner, changed = _skeletonize_source(segment.inner, canonical)
    except Exception:
        logger.warning(
            "context_compression: skeletonizer raised for language=%s; keeping original",
            segment.language,
            exc_info=True,
        )
        return segment, False

    if not changed:
        return segment, False

    new_text = (segment.fence_open or "") + new_inner + (segment.fence_close or "")
    return replace(segment, text=new_text, inner=new_inner), True
