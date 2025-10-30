#!/usr/bin/env python3
"""Generar variaciones sintéticas adicionales para el dataset PIA.

Usage:
    python agents/pia/scripts/augment_dataset.py
    python agents/pia/scripts/augment_dataset.py \
        --input data/processed/pia/pia_features.csv \
        --output data/processed/pia/pia_features_augmented.csv

Une baselines históricos y replica observaciones con ruido controlado para enriquecer entrenamiento."""

from __future__ import annotations

import argparse
from pathlib import Path

import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.loader import PROJECT_ROOT, get_path
from config.metadata import write_metadata

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.pia.src.simulator import augment_dataframe, save_dataframe

DEFAULT_INPUT_PATH = get_path('data', 'processed', 'pia', 'features')
DEFAULT_OUTPUT_PATH = get_path('data', 'processed', 'pia', 'features_augmented')
DEFAULT_BASELINES_PATH = get_path('data', 'processed', 'pia', 'baselines')


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ampliar dataset PIA con variaciones sintéticas")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--baselines",
        type=Path,
        default=DEFAULT_BASELINES_PATH,
        help="CSV con baselines históricos de consumo GNV",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    input_path = Path(args.input)
    baselines_path = Path(args.baselines)
    if not input_path.exists():
        raise SystemExit(f"Dataset base no encontrado: {input_path}")

    df = pd.read_csv(input_path)
    if baselines_path.exists():
        baselines = pd.read_csv(baselines_path)
        baseline_cols = [c for c in baselines.columns if c != "placa"]
        df = df.merge(baselines, on="placa", how="left", suffixes=("", "_baseline"))
        print(
            "Baselines de consumo integrados",
            f"{len(baselines)} placas, columnas añadidas: {baseline_cols[:5]}..."
            if baseline_cols
            else "sin columnas nuevas",
        )
    else:
        print(f"Baselines no encontrados en {baselines_path}, se continúa sin enriquecimiento")
    augmented = augment_dataframe(df, rng)
    output_path = Path(args.output)
    save_dataframe(augmented, output_path)
    print(f"Dataset ampliado guardado en {_rel(output_path)} ({len(augmented)} filas)")
    inputs = [input_path]
    if baselines_path.exists():
        inputs.append(baselines_path)
    write_metadata(
        output_path,
        script=__file__,
        inputs=inputs,
        extra={"rows": int(len(augmented)), "columns": int(len(augmented.columns))}
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
