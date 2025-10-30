"""Utility helpers to load shared configuration for scripts and services."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CONFIG_DIR.parent
DEFAULTS_FILE = CONFIG_DIR / "defaults.yaml"

class ConfigError(RuntimeError):
    """Raised when a required configuration value is missing."""


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"Config file {path} must contain a mapping at top level")
    return data


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    """Return defaults merged with local overrides."""
    config = _load_yaml(DEFAULTS_FILE)
    local_name = os.getenv("RAG_CONFIG_LOCAL", "local.yaml")
    local_file = CONFIG_DIR / local_name
    if local_file.exists():
        config = _deep_merge(config, _load_yaml(local_file))
    return config


def get_config(*keys: str, default: Any = None, required: bool = False) -> Any:
    """Fetch a configuration value using nested keys."""
    node: Any = load_config()
    for key in keys:
        if isinstance(node, dict) and key in node:
            node = node[key]
        else:
            if required and default is None:
                joined = ".".join(keys)
                raise ConfigError(f"Missing configuration key: {joined}")
            return default
    return node


def _resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path


def get_path(*keys: str, default: str | Path | None = None, required: bool = True) -> Path | None:
    """Return a project-absolute Path for the given config keys."""
    value = get_config(*keys, default=default, required=required and default is None)
    if value is None:
        if required:
            joined = ".".join(keys)
            raise ConfigError(f"Missing configuration path for {joined}")
        return None
    return _resolve_path(value)


def as_dict() -> dict[str, Any]:
    """Return a deep copy of the loaded configuration."""
    import copy

    return copy.deepcopy(load_config())
