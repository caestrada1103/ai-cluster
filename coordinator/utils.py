def parse_env_int(key: str, default: int = 0) -> int:
    """Return integer env var or default."""
    import os
    return int(os.getenv(key, default))