"""Tests for coordinator.config — Settings defaults and validators."""

import pytest
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from coordinator.config import DiscoveryMethod, Settings


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

def test_defaults():
    with patch.dict("os.environ", {}, clear=False):
        s = Settings(_env_file=None)
    assert s.host == "0.0.0.0"
    assert s.port == 8000
    assert s.discovery_method == DiscoveryMethod.STATIC
    assert s.enable_auth is False
    assert s.enable_batching is True
    assert s.max_batch_size == 32
    assert s.request_timeout == 300


def test_custom_port():
    s = Settings(port=9000, _env_file=None)
    assert s.port == 9000


def test_port_validation_zero_raises():
    with pytest.raises(ValidationError):
        Settings(port=0, _env_file=None)


def test_port_validation_too_high_raises():
    with pytest.raises(ValidationError):
        Settings(port=70000, _env_file=None)


# ---------------------------------------------------------------------------
# static_workers validator
# ---------------------------------------------------------------------------

def test_static_workers_from_comma_string():
    s = Settings(static_workers="host1:50051,host2:50052", _env_file=None)
    assert s.static_workers == ["host1:50051", "host2:50052"]


def test_static_workers_trims_whitespace():
    s = Settings(static_workers=" host1:50051 , host2:50052 ", _env_file=None)
    assert s.static_workers == ["host1:50051", "host2:50052"]


def test_static_workers_from_list():
    s = Settings(static_workers=["host1:50051", "host2:50052"], _env_file=None)
    assert s.static_workers == ["host1:50051", "host2:50052"]


def test_static_workers_empty_string_gives_empty_list():
    s = Settings(static_workers="", _env_file=None)
    assert s.static_workers == []


# ---------------------------------------------------------------------------
# load_models_config
# ---------------------------------------------------------------------------

def test_load_models_config_missing_file(tmp_path):
    s = Settings(models_config=tmp_path / "nonexistent.toml", _env_file=None)
    result = s.load_models_config()
    assert result == {}


def test_load_models_config_toml(tmp_path):
    toml_file = tmp_path / "models.toml"
    toml_file.write_text('[models]\n[models.test]\nname = "test"\n')
    s = Settings(models_config=toml_file, _env_file=None)
    result = s.load_models_config()
    assert "models" in result


def test_load_models_config_unsupported_extension(tmp_path):
    bad_file = tmp_path / "models.json"
    bad_file.write_text("{}")
    s = Settings(models_config=bad_file, _env_file=None)
    with pytest.raises(ValueError, match="Unsupported"):
        s.load_models_config()
