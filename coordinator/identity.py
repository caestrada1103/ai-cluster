"""Key -> caller identity resolution.

Callers come from ``COORDINATOR_API_KEYS`` (flat, all-admin, unrestricted --
today's behavior) and/or ``COORDINATOR_API_KEY_FILE`` (a TOML file assigning
roles/model restrictions per key). Both are read live from the environment on
each call, matching auth.load_api_keys()'s no-caching contract. See
docs/configuration.md.
"""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, FrozenSet, List, Mapping, Optional, Tuple

import tomllib

_ROLES: FrozenSet[str] = frozenset({"admin", "user"})


@dataclass(frozen=True)
class Caller:
    """A resolved identity for a validated API key."""

    id: str
    role: str
    models: FrozenSet[str]

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"

    def may_use_model(self, model_name: str) -> bool:
        """True when unrestricted (empty `models`) or `model_name` is listed."""
        return not self.models or model_name in self.models


#: Used when no keys are configured at all (auth off) -- preserves today's
#: fully-open behavior.
UNRESTRICTED = Caller(id="anonymous", role="admin", models=frozenset())

#: Parsed-file cache keyed on (path, mtime_ns, size) so a request doesn't
#: re-parse TOML every call, but an edited file is picked up without a
#: restart.
_file_cache: Dict[Tuple[str, int, int], Dict[str, Caller]] = {}


def _load_flat_keys_ordered() -> List[str]:
    """Parse COORDINATOR_API_KEYS preserving declared order (dedup, first wins)."""
    raw = os.environ.get("COORDINATOR_API_KEYS", "")
    ordered: List[str] = []
    seen: set[str] = set()
    for key in raw.split(","):
        key = key.strip()
        if key and key not in seen:
            seen.add(key)
            ordered.append(key)
    return ordered


def _parse_file(path: Path) -> Dict[str, Caller]:
    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except OSError as exc:
        raise ValueError(f"COORDINATOR_API_KEY_FILE {path}: cannot read file: {exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"COORDINATOR_API_KEY_FILE {path}: invalid TOML: {exc}") from exc

    unknown_tables = set(data) - {"keys"}
    if unknown_tables:
        raise ValueError(
            f"COORDINATOR_API_KEY_FILE {path}: unknown top-level table(s) {sorted(unknown_tables)}"
        )

    callers: Dict[str, Caller] = {}
    seen_keys: Dict[str, str] = {}
    for label, entry in data.get("keys", {}).items():
        if not isinstance(entry, dict):
            raise ValueError(f"COORDINATOR_API_KEY_FILE {path}: [keys.{label}] must be a table")

        key = entry.get("key")
        if not isinstance(key, str) or not key.strip():
            raise ValueError(
                f"COORDINATOR_API_KEY_FILE {path}: [keys.{label}].key must be a non-empty string"
            )
        key = key.strip()

        role = entry.get("role", "user")
        if role not in _ROLES:
            raise ValueError(
                f"COORDINATOR_API_KEY_FILE {path}: [keys.{label}].role must be one of "
                f"{sorted(_ROLES)}, got {role!r}"
            )

        models = entry.get("models", [])
        if not isinstance(models, list) or not all(
            isinstance(m, str) and m.strip() for m in models
        ):
            raise ValueError(
                f"COORDINATOR_API_KEY_FILE {path}: [keys.{label}].models must be a list of "
                "non-empty strings"
            )

        if key in seen_keys:
            raise ValueError(
                f"COORDINATOR_API_KEY_FILE {path}: key duplicated between "
                f"[keys.{seen_keys[key]}] and [keys.{label}]"
            )
        seen_keys[key] = label

        callers[key] = Caller(id=label, role=role, models=frozenset(models))

    return callers


def _load_file_keys() -> Optional[Dict[str, Caller]]:
    """Load and cache `COORDINATOR_API_KEY_FILE`; None when the var is unset."""
    raw_path = os.environ.get("COORDINATOR_API_KEY_FILE", "")
    if not raw_path.strip():
        return None
    path = Path(raw_path.strip())
    try:
        stat = path.stat()
    except OSError as exc:
        raise ValueError(f"COORDINATOR_API_KEY_FILE {path}: cannot read file: {exc}") from exc

    cache_key = (str(path), stat.st_mtime_ns, stat.st_size)
    cached = _file_cache.get(cache_key)
    if cached is not None:
        return cached

    parsed = _parse_file(path)
    _file_cache.clear()  # only ever one file configured at a time
    _file_cache[cache_key] = parsed
    return parsed


def load_identities() -> Mapping[str, Caller]:
    """Build the full key -> Caller table from both config sources.

    No file configured: every flat key becomes an admin/unrestricted
    Caller ("key-1", "key-2", ... by declared order) -- byte-for-byte
    today's behavior. File configured: file entries define identity; a
    flat-only key gets role="user"/unrestricted; a key in both uses the
    file entry.
    """
    flat_keys = _load_flat_keys_ordered()
    file_callers = _load_file_keys()

    if file_callers is None:
        return {
            key: Caller(id=f"key-{i}", role="admin", models=frozenset())
            for i, key in enumerate(flat_keys, start=1)
        }

    identities: Dict[str, Caller] = {
        key: Caller(id=f"key-{i}", role="user", models=frozenset())
        for i, key in enumerate(flat_keys, start=1)
    }
    identities.update(file_callers)
    return identities


def has_declared_identities() -> bool:
    """True when `COORDINATOR_API_KEY_FILE` is set to a non-empty value."""
    return bool(os.environ.get("COORDINATOR_API_KEY_FILE", "").strip())


def resolve_caller(candidate: str) -> Optional[Caller]:
    """Constant-time lookup of `candidate` against every configured identity.

    Compares UTF-8 bytes (not str) via secrets.compare_digest, iterating
    every entry with no early return, keeping the last match -- mirrors
    auth._matches_any so a non-ASCII candidate 401s instead of raising.
    """
    candidate_bytes = candidate.encode("utf-8")
    matched: Optional[Caller] = None
    for key, caller in load_identities().items():
        if secrets.compare_digest(candidate_bytes, key.encode("utf-8")):
            matched = caller
    return matched
