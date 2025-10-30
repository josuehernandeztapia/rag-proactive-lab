#!/usr/bin/env python3
"""Generar reporte rápido de escenarios PIA a partir del dataset.

Usage:
    python agents/pia/scripts/report_scenarios.py
    python agents/pia/scripts/report_scenarios.py \
        --input data/processed/pia/pia_features_augmented.csv \
        --output reports/pia_scenarios_summary.csv

Resume escenarios, segmentos y métricas agregadas para revisión humana."""

from __future__ import annotations

import argparse
from pathlib import Path

import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.loader import get_path
from config.metadata import write_metadata

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.pia.src.simulator import save_dataframe, summarize_scenarios

DEFAULT_INPUT_PATH = get_path('data', 'processed', 'pia', 'features_augmented')
DEFAULT_OUTPUT_PATH = get_path('reports', 'pia_scenarios_summary')


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resumen de escenarios PIA")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Dataset no encontrado: {_rel(input_path)}")
    df = pd.read_csv(input_path)
    summary = summarize_scenarios(df)
    output_path = Path(args.output)
    save_dataframe(summary, output_path)
    print(f"Resumen guardado en {_rel(output_path)} ({len(summary)} filas)")
    write_metadata(
        output_path,
        script=__file__,
        inputs=[input_path],
        extra={"rows": int(len(summary)), "columns": int(len(summary.columns))}
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
