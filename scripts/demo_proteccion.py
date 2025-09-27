#!/usr/bin/env python3
"""Orchestrates the synthetic protection demo end-to-end."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: Sequence[str], *, env: dict[str, str] | None = None) -> None:
    display = " ".join(cmd)
    print(f"→ {display}")
    subprocess.run(cmd, check=True, cwd=ROOT, env=env)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the synthetic protection demo pipeline")
    parser.add_argument("--size", type=int, default=200, help="Cantidad de financiamientos sintéticos a generar")
    parser.add_argument("--seed", type=int, default=2025, help="Semilla para reproducibilidad")
    parser.add_argument("--skip-seed", action="store_true", help="No regenerar cartera antes de la corrida")
    parser.add_argument("--skip-monitor", action="store_true", help="Omitir el monitor CLI al final")
    parser.add_argument("--llm", action="store_true", help="Ejecutar pia_llm_notifier en modo plantilla al finalizar")
    parser.add_argument("--llm-limit", type=int, default=5, help="Máximo de alertas a generar si --llm está activo")
    parser.add_argument(
        "--llm-outbox",
        type=Path,
        default=ROOT / "reports" / "pia_llm_outbox.jsonl",
        help="Archivo JSONL donde se encolan las alertas LLM",
    )
    return parser.parse_args()


def run_demo(args: argparse.Namespace) -> None:
    if not args.skip_seed:
        _run(
            [
                "python3",
                "scripts/pia_seed_synthetic_portfolio.py",
                "--size",
                str(args.size),
                "--seed",
                str(args.seed),
            ]
        )
    else:
        print("→ Omitiendo generación de cartera sintética (--skip-seed)")

    _run(["python3", "scripts/pia_generate_dummy_outcomes.py", "--reset-log"])

    if not args.skip_monitor:
        _run(["python3", "scripts/pia_plan_summary_monitor.py"])
    else:
        print("→ Omitiendo monitor CLI (--skip-monitor)")

    if args.llm:
        env = os.environ.copy()
        env.setdefault("PIA_LLM_MODE", "template")
        env.setdefault("PIA_LLM_ALERTS", "1")
        cmd = [
            "python3",
            "scripts/pia_llm_notifier.py",
            "--limit",
            str(args.llm_limit),
            "--pia-outbox",
            str(args.llm_outbox),
            "--skip-email",
        ]
        _run(cmd, env=env)
        print(f"Alertas guardadas en {args.llm_outbox}")
    else:
        print("→ Notifier LLM omitido (usa --llm para habilitarlo)")

    print("\nDemo completada. Artefactos clave:")
    print("  - data/pia/synthetic_driver_states.csv")
    print("  - data/pia/pia_outcomes_log.csv")
    print("  - data/hase/pia_outcomes_features.csv")
    print("  - reports/pia_plan_summary.csv")


def main() -> int:
    try:
        run_demo(parse_args())
    except subprocess.CalledProcessError as exc:
        print(f"Comando falló con código {exc.returncode}", file=sys.stderr)
        return exc.returncode or 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
