"""pytest configuration and shared helpers for coordinator tests."""

from typing import Any

from coordinator.config import Settings


def make_settings(**overrides: Any) -> Settings:
    """Build Settings isolated from any real .env file on the dev machine."""
    return Settings(_env_file=None, **overrides)
