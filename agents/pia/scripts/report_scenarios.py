#!/usr/bin/env python3
"""Generar reporte rápido de escenarios PIA a partir del dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import sys
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from agents.pia.src.simulator import save_dataframe, summarize_scenarios


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resumen de escenarios PIA")
    parser.add_argument("--input", type=Path, default=Path("data/pia/pia_features_augmented.csv"))
    parser.add_argument("--output", type=Path, default=Path("reports/pia_scenarios_summary.csv"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input.exists():
        raise SystemExit(f"Dataset no encontrado: {args.input}")
    df = pd.read_csv(args.input)
    summary = summarize_scenarios(df)
    save_dataframe(summary, args.output)
    print(f"Resumen guardado en {args.output} ({len(summary)} filas)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
