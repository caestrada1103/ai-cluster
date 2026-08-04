"""Tests for coordinator.identity -- key -> caller identity resolution.

Both COORDINATOR_API_KEYS and COORDINATOR_API_KEY_FILE are read live from the
environment on every call (no cache to reset between tests, beyond the
file-parse cache exercised explicitly below).
"""

import os
from pathlib import Path

import pytest

from coordinator import identity


@pytest.fixture(autouse=True)
def _clear_identity_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("COORDINATOR_API_KEYS", raising=False)
    monkeypatch.delenv("COORDINATOR_API_KEY_FILE", raising=False)
    identity._file_cache.clear()


def _write_toml(path: Path, text: str) -> None:
    path.write_text(text)


# ---------------------------------------------------------------------------
# Flat list only
# ---------------------------------------------------------------------------


def test_flat_list_only_assigns_ids_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COORDINATOR_API_KEYS", "key-aaa, key-bbb, key-ccc")
    identities = identity.load_identities()
    assert identities["key-aaa"] == identity.Caller(id="key-1", role="admin", models=frozenset())
    assert identities["key-bbb"] == identity.Caller(id="key-2", role="admin", models=frozenset())
    assert identities["key-ccc"] == identity.Caller(id="key-3", role="admin", models=frozenset())


def test_flat_list_callers_are_admin_and_unrestricted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COORDINATOR_API_KEYS", "solo-key")
    caller = identity.load_identities()["solo-key"]
    assert caller.is_admin
    assert caller.may_use_model("anything-at-all")


def test_no_keys_configured_returns_empty_mapping() -> None:
    assert dict(identity.load_identities()) == {}


# ---------------------------------------------------------------------------
# File only
# ---------------------------------------------------------------------------


def test_file_only_parses_ids_roles_and_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    toml_path = tmp_path / "keys.toml"
    _write_toml(
        toml_path,
        """
        [keys.ci-runner]
        key = "ci-secret"
        role = "user"
        models = ["qwen2.5-0.5b-gguf"]

        [keys.ops]
        key = "ops-secret"
        role = "admin"
        """,
    )
    monkeypatch.setenv("COORDINATOR_API_KEY_FILE", str(toml_path))

    identities = identity.load_identities()

    ci = identities["ci-secret"]
    assert ci.id == "ci-runner"
    assert ci.role == "user"
    assert not ci.is_admin
    assert ci.models == frozenset({"qwen2.5-0.5b-gguf"})
    assert ci.may_use_model("qwen2.5-0.5b-gguf")
    assert not ci.may_use_model("some-other-model")

    ops = identities["ops-secret"]
    assert ops.id == "ops"
    assert ops.is_admin
    assert ops.models == frozenset()
    assert ops.may_use_model("anything")


def test_file_models_omitted_means_unrestricted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    toml_path = tmp_path / "keys.toml"
    _write_toml(
        toml_path,
        """
        [keys.plain]
        key = "plain-secret"
        """,
    )
    monkeypatch.setenv("COORDINATOR_API_KEY_FILE", str(toml_path))

    caller = identity.load_identities()["plain-secret"]
    assert caller.role == "user"
    assert caller.models == frozenset()
    assert caller.may_use_model("whatever")


def test_file_default_role_is_user(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    toml_path = tmp_path / "keys.toml"
    _write_toml(toml_path, '[keys.a]\nkey = "a-secret"\n')
    monkeypatch.setenv("COORDINATOR_API_KEY_FILE", str(toml_path))
    assert identity.load_identities()["a-secret"].role == "user"


# ---------------------------------------------------------------------------
# Both set: precedence
# ---------------------------------------------------------------------------


def test_both_set_file_wins_for_shared_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    toml_path = tmp_path / "keys.toml"
    _write_toml(
        toml_path,
        """
        [keys.shared-caller]
        key = "shared-secret"
        role = "admin"
        models = ["only-this-model"]
        """,
    )
    monkeypatch.setenv("COORDINATOR_API_KEY_FILE", str(toml_path))
    monkeypatch.setenv("COORDINATOR_API_KEYS", "shared-secret, flat-only-secret")

    identities = identity.load_identities()

    shared = identities["shared-secret"]
    assert shared.id == "shared-caller"
    assert shared.role == "admin"
    assert shared.models == frozenset({"only-this-model"})

    flat_only = identities["flat-only-secret"]
    assert flat_only.role == "user"
    assert flat_only.models == frozenset()
    assert not flat_only.is_admin


def test_configured_but_empty_file_still_demotes_flat_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Configuring the file at all means roles are declared there, so a flat
    key that is absent from it is not an admin."""
    toml_path = tmp_path / "keys.toml"
    _write_toml(toml_path, "[keys]\n")
    monkeypatch.setenv("COORDINATOR_API_KEY_FILE", str(toml_path))
    monkeypatch.setenv("COORDINATOR_API_KEYS", "flat-only-secret")

    caller = identity.load_identities()["flat-only-secret"]
    assert caller.role == "user"
    assert not caller.is_admin


# ---------------------------------------------------------------------------
# Validation -- fail closed with ValueError
# ---------------------------------------------------------------------------


def _set_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, text: str) -> None:
    toml_path = tmp_path / "keys.toml"
    _write_toml(toml_path, text)
    monkeypatch.setenv("COORDINATOR_API_KEY_FILE", str(toml_path))


def test_unreadable_file_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("COORDINATOR_API_KEY_FILE", str(tmp_path / "does-not-exist.toml"))
    with pytest.raises(ValueError):
        identity.load_identities()


def test_unparseable_toml_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _set_file(monkeypatch, tmp_path, "this is not [ valid toml")
    with pytest.raises(ValueError):
        identity.load_identities()


def test_missing_key_field_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _set_file(monkeypatch, tmp_path, '[keys.a]\nrole = "admin"\n')
    with pytest.raises(ValueError):
        identity.load_identities()


def test_non_string_key_field_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _set_file(monkeypatch, tmp_path, "[keys.a]\nkey = 12345\n")
    with pytest.raises(ValueError):
        identity.load_identities()


def test_empty_key_after_strip_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _set_file(monkeypatch, tmp_path, '[keys.a]\nkey = "   "\n')
    with pytest.raises(ValueError):
        identity.load_identities()


def test_invalid_role_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _set_file(monkeypatch, tmp_path, '[keys.a]\nkey = "a-secret"\nrole = "superuser"\n')
    with pytest.raises(ValueError):
        identity.load_identities()


def test_models_not_a_list_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _set_file(monkeypatch, tmp_path, '[keys.a]\nkey = "a-secret"\nmodels = "not-a-list"\n')
    with pytest.raises(ValueError):
        identity.load_identities()


def test_models_with_empty_string_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _set_file(monkeypatch, tmp_path, '[keys.a]\nkey = "a-secret"\nmodels = ["ok", ""]\n')
    with pytest.raises(ValueError):
        identity.load_identities()


def test_models_with_non_string_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _set_file(monkeypatch, tmp_path, '[keys.a]\nkey = "a-secret"\nmodels = ["ok", 5]\n')
    with pytest.raises(ValueError):
        identity.load_identities()


def test_duplicate_key_across_labels_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _set_file(
        monkeypatch,
        tmp_path,
        """
        [keys.a]
        key = "same-secret"

        [keys.b]
        key = "same-secret"
        """,
    )
    with pytest.raises(ValueError):
        identity.load_identities()


def test_unknown_top_level_table_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _set_file(monkeypatch, tmp_path, '[stuff]\nfoo = "bar"\n')
    with pytest.raises(ValueError):
        identity.load_identities()


# ---------------------------------------------------------------------------
# resolve_caller
# ---------------------------------------------------------------------------


def test_resolve_caller_returns_none_for_unknown_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COORDINATOR_API_KEYS", "known-key")
    assert identity.resolve_caller("unknown-key") is None


def test_resolve_caller_does_not_raise_on_non_ascii_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("COORDINATOR_API_KEYS", "known-key")
    assert identity.resolve_caller("café") is None


def test_resolve_caller_returns_matching_caller(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COORDINATOR_API_KEYS", "known-key")
    caller = identity.resolve_caller("known-key")
    assert caller is not None
    assert caller.id == "key-1"


# ---------------------------------------------------------------------------
# File-change / caching behavior
# ---------------------------------------------------------------------------


def test_changed_file_is_picked_up_without_restart(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    toml_path = tmp_path / "keys.toml"
    _write_toml(toml_path, '[keys.a]\nkey = "secret-v1"\n')
    monkeypatch.setenv("COORDINATOR_API_KEY_FILE", str(toml_path))
    assert "secret-v1" in identity.load_identities()

    _write_toml(toml_path, '[keys.a]\nkey = "secret-v2"\n')
    # Force a distinguishable mtime in case the filesystem clock granularity
    # would otherwise make the write above indistinguishable from the first.
    stat = toml_path.stat()
    os.utime(toml_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))

    identities = identity.load_identities()
    assert "secret-v2" in identities
    assert "secret-v1" not in identities


def test_unchanged_file_is_not_reparsed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    toml_path = tmp_path / "keys.toml"
    _write_toml(toml_path, '[keys.a]\nkey = "secret-v1"\n')
    monkeypatch.setenv("COORDINATOR_API_KEY_FILE", str(toml_path))

    parse_calls = []
    original_parse = identity._parse_file

    def _counting_parse(path: Path) -> "dict[str, identity.Caller]":
        parse_calls.append(path)
        return original_parse(path)

    monkeypatch.setattr(identity, "_parse_file", _counting_parse)

    identity.load_identities()
    identity.load_identities()
    identity.load_identities()

    assert len(parse_calls) == 1
