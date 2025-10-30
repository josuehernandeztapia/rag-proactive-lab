"""Helpers to emit metadata.json files next to generated artifacts."""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Any

from .loader import PROJECT_ROOT, get_config


def _normalize_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _current_commit() -> str | None:
    include = bool(get_config("metadata", "include_git_commit", default=True))
    if not include:
        return None
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True)
        return out.strip()
    except Exception:
        return None


def write_metadata(
    artifact: str | Path,
    *,
    script: str | Path,
    inputs: Iterable[str | Path] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Path | None:
    """Create a `*.metadata.json` file alongside the artifact.

    Returns the path to the metadata file or ``None`` if metadata emission is disabled.
    """
    if not bool(get_config("metadata", "enabled", default=True)):
        return None

    artifact_path = _normalize_path(artifact)
    metadata_path = artifact_path.parent / f"{artifact_path.name}.metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "artifact": _relative(artifact_path),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "script": _relative(_normalize_path(script)),
    }

    commit = _current_commit()
    if commit:
        payload["commit"] = commit

    if inputs:
        payload["inputs"] = sorted({_relative(_normalize_path(p)) for p in inputs})

    if extra:
        for key, value in extra.items():
            if isinstance(value, Path):
                payload[key] = _relative(value)
            else:
                payload[key] = value

    metadata_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return metadata_path
