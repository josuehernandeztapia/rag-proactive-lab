#!/usr/bin/env python3
"""Watcher simple para disparar hase_llm_notifier cuando cambie el CSV de enhanced default predictions.

Modo de uso:
    python3 agents/hase/scripts/hase_llm_worker.py \
        --features data/processed/hase/enhanced_default_predictions.csv \
        --interval 60 \
        --notifier-args "--limit 3 --min-default-risk 0.7"

Se ejecuta en bucle infinito (Ctrl+C para detener).
Si todo lo que necesitas es un cron, puedes usar directamente hase_llm_notifier.py.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path

from config.loader import PROJECT_ROOT, get_path

DEFAULT_FEATURES = get_path('data', 'processed', 'hase', 'enhanced_default_predictions')
DEFAULT_INTERVAL = 60


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watcher que dispara hase_llm_notifier al detectar cambios")
    parser.add_argument(
        "--features",
        type=Path,
        default=DEFAULT_FEATURES,
        help="Ruta del CSV que se monitorea (default data/processed/hase/enhanced_default_predictions.csv)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=DEFAULT_INTERVAL,
        help="Segundos entre revisiones (default 60)",
    )
    parser.add_argument(
        "--notifier-args",
        default="",
        help="Argumentos adicionales para pasar a hase_llm_notifier.py (ej. '--limit 3 --min-default-risk 0.8')",
    )
    parser.add_argument(
        "--oneshot",
        action="store_true",
        help="Ejecuta una sola vez (sin loop). Útil para integración con otras herramientas.",
    )
    return parser.parse_args()


def run_notifier(extra_args: str) -> int:
    notifier_path = PROJECT_ROOT / "agents" / "hase" / "scripts" / "hase_llm_notifier.py"
    cmd = [sys.executable, str(notifier_path)]
    if extra_args:
        cmd.extend(shlex.split(extra_args))
    process = subprocess.run(cmd, capture_output=False)
    return process.returncode


def worker(features: Path, interval: int, extra_args: str, oneshot: bool) -> None:
    last_mtime = None
    if not features.exists():
        print(f"[hase_llm_worker] El archivo {_rel(features)} no existe todavía; esperando...", flush=True)
    while True:
        if features.exists():
            current_mtime = features.stat().st_mtime
            if last_mtime is None or current_mtime > last_mtime:
                print(f"[hase_llm_worker] Cambio detectado en {_rel(features)}; disparando hase_llm_notifier", flush=True)
                exit_code = run_notifier(extra_args)
                if exit_code != 0:
                    print(f"[hase_llm_worker] hase_llm_notifier devolvió código {exit_code}", flush=True)
                last_mtime = current_mtime
                if oneshot:
                    break
        else:
            print(f"[hase_llm_worker] {_rel(features)} aún no existe...", flush=True)
            last_mtime = None
        time.sleep(max(5, interval))


def main() -> int:
    args = parse_args()
    try:
        worker(args.features, args.interval, args.notifier_args, args.oneshot)
    except KeyboardInterrupt:
        print("[hase_llm_worker] Detenido por el usuario.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())