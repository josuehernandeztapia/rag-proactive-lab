#!/usr/bin/env python3
"""Eliminar artefactos efímeros generados durante demos/tests.

El script borra:
- Logs temporales (`.ngrok*`, `.uvicorn*`, `*.pid`).
- Archivos de cola (`*.processing`, `logs/*.jsonl` vacíos).
- Directorios `logs/` y `reports/` vacíos.

No toca datasets ni archivos de configuración. Ejecuta en la raíz del repo.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[2]

GLOB_PATTERNS: Iterable[str] = (
    ".ngrok*",
    ".uvicorn*",
    "*.pid",
    "logs/*.processing",
)

EMPTY_LOG_TARGETS: Iterable[Path] = (
    ROOT / "logs" / "events.jsonl",
    ROOT / "logs" / "media_queue.jsonl",
)


def remove_globs() -> list[Path]:
    removed: list[Path] = []
    for pattern in GLOB_PATTERNS:
        for path in ROOT.glob(pattern):
            try:
                path.unlink()
                removed.append(path)
            except FileNotFoundError:
                continue
    return removed


def trim_empty_logs() -> list[Path]:
    trimmed: list[Path] = []
    for target in EMPTY_LOG_TARGETS:
        if target.exists() and target.stat().st_size == 0:
            target.unlink()
            trimmed.append(target)
    return trimmed


def prune_empty_dirs() -> list[Path]:
    pruned: list[Path] = []
    for folder in (ROOT / "logs", ROOT / "reports"):
        if folder.exists():
            try:
                next(folder.iterdir())
            except StopIteration:
                folder.rmdir()
                pruned.append(folder)
    return pruned


def main() -> int:
    parser = argparse.ArgumentParser(description="Cleanup ephemereal artefacts")
    parser.add_argument("--dry-run", action="store_true", help="Solo reporta, no elimina")
    args = parser.parse_args()

    removed = remove_globs() if not args.dry_run else []
    trimmed = trim_empty_logs() if not args.dry_run else []
    pruned = prune_empty_dirs() if not args.dry_run else []

    if args.dry_run:
        print("Dry run: no se eliminó nada. Objetivos potenciales:")
        for pattern in GLOB_PATTERNS:
            matches = list(ROOT.glob(pattern))
            for match in matches:
                print(f"  {match.relative_to(ROOT)}")
        for target in EMPTY_LOG_TARGETS:
            print(f"  {target.relative_to(ROOT)} (si está vacío)")
        return 0

    for path in removed:
        print(f"Removido {path.relative_to(ROOT)}")
    for path in trimmed:
        print(f"Eliminado log vacío {path.relative_to(ROOT)}")
    for path in pruned:
        print(f"Directorio vacío eliminado {path.relative_to(ROOT)}")
    if not (removed or trimmed or pruned):
        print("No había artefactos temporales que limpiar.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
