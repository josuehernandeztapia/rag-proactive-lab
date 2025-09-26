#!/usr/bin/env python3
"""Generar variaciones sintéticas adicionales para el dataset PIA."""

from __future__ import annotations

import argparse
from pathlib import Path

import sys
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from agents.pia.src.simulator import augment_dataframe, save_dataframe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ampliar dataset PIA con variaciones sintéticas")
    parser.add_argument("--input", type=Path, default=Path("data/pia/pia_features.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/pia/pia_features_augmented.csv"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--baselines",
        type=Path,
        default=Path("data/pia/hase_consumption_baselines.csv"),
        help="CSV con baselines históricos de consumo GNV",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    if not args.input.exists():
        raise SystemExit(f"Dataset base no encontrado: {args.input}")

    df = pd.read_csv(args.input)
    if args.baselines.exists():
        baselines = pd.read_csv(args.baselines)
        baseline_cols = [c for c in baselines.columns if c != "placa"]
        df = df.merge(baselines, on="placa", how="left", suffixes=("", "_baseline"))
        print(
            "Baselines de consumo integrados",
            f"{len(baselines)} placas, columnas añadidas: {baseline_cols[:5]}..."
            if baseline_cols
            else "sin columnas nuevas",
        )
    else:
        print(f"Baselines no encontrados en {args.baselines}, se continúa sin enriquecimiento")
    augmented = augment_dataframe(df, rng)
    save_dataframe(augmented, args.output)
    print(f"Dataset ampliado guardado en {args.output} ({len(augmented)} filas)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
