"""Tests for coordinator.config — Settings defaults and validators."""

from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from coordinator.config import DiscoveryMethod, Settings
from coordinator.tests.conftest import make_settings

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_defaults() -> None:
    with patch.dict("os.environ", {}, clear=False):
        s = make_settings()
    assert s.host == "0.0.0.0"
    assert s.port == 8000
    assert s.discovery_method == DiscoveryMethod.STATIC
    assert s.request_timeout == 300


def test_custom_port() -> None:
    s = make_settings(port=9000)
    assert s.port == 9000


def test_port_validation_zero_raises() -> None:
    with pytest.raises(ValidationError):
        make_settings(port=0)


def test_port_validation_too_high_raises() -> None:
    with pytest.raises(ValidationError):
        make_settings(port=70000)


# ---------------------------------------------------------------------------
# static_workers validator
# ---------------------------------------------------------------------------


def test_static_workers_from_comma_string() -> None:
    s = make_settings(static_workers="host1:50051,host2:50052")
    assert s.static_workers == ["host1:50051", "host2:50052"]


def test_static_workers_trims_whitespace() -> None:
    s = make_settings(static_workers=" host1:50051 , host2:50052 ")
    assert s.static_workers == ["host1:50051", "host2:50052"]


def test_static_workers_from_list() -> None:
    s = make_settings(static_workers=["host1:50051", "host2:50052"])
    assert s.static_workers == ["host1:50051", "host2:50052"]


def test_static_workers_empty_string_gives_empty_list() -> None:
    s = make_settings(static_workers="")
    assert s.static_workers == []


# ---------------------------------------------------------------------------
# load_models_config
# ---------------------------------------------------------------------------


def test_load_models_config_missing_file(tmp_path: Path) -> None:
    s = make_settings(models_config=tmp_path / "nonexistent.toml")
    result = s.load_models_config()
    assert result == {}


def test_load_models_config_toml(tmp_path: Path) -> None:
    toml_file = tmp_path / "models.toml"
    toml_file.write_text('[models]\n[models.test]\nname = "test"\n')
    s = make_settings(models_config=toml_file)
    result = s.load_models_config()
    assert "models" in result


def test_load_models_config_unsupported_extension(tmp_path: Path) -> None:
    bad_file = tmp_path / "models.json"
    bad_file.write_text("{}")
    s = make_settings(models_config=bad_file)
    with pytest.raises(ValueError, match="Unsupported"):
        s.load_models_config()


def test_settings_ignores_unknown_dotenv_keys(tmp_path: Path) -> None:
    """The shared .env ships worker vars (GPU_COUNT, HF_TOKEN, ...); Settings must ignore them."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "GPU_COUNT=1\nHF_TOKEN=hf_dummy\nRUST_LOG=info\nGPU_INDEX=0\n" "COORDINATOR_PORT=8123\n"
    )
    s = Settings(_env_file=str(env_file))
    assert s.port == 8123


def test_cors_origins_star_from_env(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("COORDINATOR_CORS_ORIGINS=*\n")
    s = Settings(_env_file=str(env_file))
    assert s.cors_origins == ["*"]


def test_cors_origins_comma_separated() -> None:
    s = make_settings(cors_origins="https://a.example,https://b.example")
    assert s.cors_origins == ["https://a.example", "https://b.example"]


def test_cors_origins_list_passthrough() -> None:
    s = make_settings(cors_origins=["https://a.example"])
    assert s.cors_origins == ["https://a.example"]


def test_cors_origins_json_array_string_from_env(tmp_path: Path) -> None:
    """Backward-compat: values written as a JSON array string still work."""
    env_file = tmp_path / ".env"
    env_file.write_text('COORDINATOR_CORS_ORIGINS=["https://a.example","https://b.example"]\n')
    s = Settings(_env_file=str(env_file))
    assert s.cors_origins == ["https://a.example", "https://b.example"]


def test_non_static_discovery_fails_fast_with_clear_error() -> None:
    """mdns/broadcast/consul are planned, not implemented — creating the
    discovery manager must raise a clear error instead of AttributeError."""
    from coordinator.discovery import WorkerDiscovery

    settings = make_settings(discovery_method="mdns")
    with pytest.raises(ValueError, match="not implemented"):
        WorkerDiscovery(settings)


def test_context_compression_defaults() -> None:
    from coordinator.tests.conftest import make_settings

    settings = make_settings()
    assert settings.context_compression_enabled is False
    assert settings.context_compression_token_budget == 8192
    assert settings.context_compression_safety_margin == 0.20
    assert settings.context_compression_active_segments == 1
    assert settings.context_compression_techniques == ["skeletonize"]
    assert settings.context_compression_summarizer_model == "qwen2.5-0.5b-gguf"
    assert settings.context_compression_llmlingua_enabled is False


def test_context_compression_techniques_rejects_unknown() -> None:
    import pytest

    from coordinator.tests.conftest import make_settings

    with pytest.raises(Exception, match="Unknown context_compression technique"):
        make_settings(context_compression_techniques="skeletonize,teleport")


def test_context_compression_techniques_accepts_comma_string() -> None:
    from coordinator.tests.conftest import make_settings

    settings = make_settings(context_compression_techniques="skeletonize,summarize")
    assert settings.context_compression_techniques == ["skeletonize", "summarize"]


def test_llamaserver_autoload_default_on() -> None:
    """Plan 13 Task 5 gate defaults to on (auto-load unloaded llamaserver models)."""
    assert make_settings().llamaserver_autoload is True


def test_llamaserver_autoload_env_override(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("COORDINATOR_LLAMASERVER_AUTOLOAD=false\n")
    s = Settings(_env_file=str(env_file))
    assert s.llamaserver_autoload is False


def test_dead_settings_fields_removed() -> None:
    s = make_settings()
    for dead in (
        "default_model",
        "enable_batching",
        "max_batch_size",
        "batch_timeout_ms",
        "enable_auth",
        "api_keys",
        "rate_limit_per_minute",
        "log_level",
        "log_format",
        "enable_metrics",
        "metrics_port",
        "auto_load_models",
        "model_cache_dir",
    ):
        assert not hasattr(s, dead), f"dead field still present: {dead}"
