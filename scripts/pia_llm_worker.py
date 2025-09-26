#!/usr/bin/env python3
"""Watcher simple para disparar pia_llm_notifier cuando cambie el CSV de features.

Modo de uso:
    python3 scripts/pia_llm_worker.py \
        --features data/hase/pia_outcomes_features.csv \
        --interval 60 \
        --notifier-args "--limit 5 --email-to laboratorio@rag.mx"

Se ejecuta en bucle infinito (Ctrl+C para detener).
Si todo lo que necesitas es un cron, puedes usar directamente pia_llm_notifier.py.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_FEATURES = Path("data/hase/pia_outcomes_features.csv")
DEFAULT_INTERVAL = 60


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watcher que dispara pia_llm_notifier al detectar cambios")
    parser.add_argument(
        "--features",
        type=Path,
        default=DEFAULT_FEATURES,
        help="Ruta del CSV que se monitorea (default data/hase/pia_outcomes_features.csv)",
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
        help="Argumentos adicionales para pasar a pia_llm_notifier.py (ej. '--limit 5 --email-to ...')",
    )
    parser.add_argument(
        "--oneshot",
        action="store_true",
        help="Ejecuta una sola vez (sin loop). Útil para integración con otras herramientas.",
    )
    return parser.parse_args()


def run_notifier(extra_args: str) -> int:
    cmd = [sys.executable, "scripts/pia_llm_notifier.py"]
    if extra_args:
        cmd.extend(shlex.split(extra_args))
    process = subprocess.run(cmd, capture_output=False)
    return process.returncode


def worker(features: Path, interval: int, extra_args: str, oneshot: bool) -> None:
    last_mtime = None
    if not features.exists():
        print(f"[pia_llm_worker] El archivo {features} no existe todavía; esperando...", flush=True)
    while True:
        if features.exists():
            current_mtime = features.stat().st_mtime
            if last_mtime is None or current_mtime > last_mtime:
                print(f"[pia_llm_worker] Cambio detectado en {features}; disparando pia_llm_notifier", flush=True)
                exit_code = run_notifier(extra_args)
                if exit_code != 0:
                    print(f"[pia_llm_worker] pia_llm_notifier devolvió código {exit_code}", flush=True)
                last_mtime = current_mtime
                if oneshot:
                    break
        else:
            print(f"[pia_llm_worker] {features} aún no existe...", flush=True)
            last_mtime = None
        time.sleep(max(5, interval))


def main() -> int:
    args = parse_args()
    try:
        worker(args.features, args.interval, args.notifier_args, args.oneshot)
    except KeyboardInterrupt:
        print("[pia_llm_worker] Detenido por el usuario.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
