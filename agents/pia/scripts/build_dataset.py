#!/usr/bin/env python3
"""Generate a synthetic PIA dataset using HASE feature store.

Usage:
    python agents/pia/scripts/build_dataset.py
    python agents/pia/scripts/build_dataset.py \
        --snapshot data/processed/hase/consumos_snapshot_latest.csv.gz \
        --target-payment 18000 --output data/processed/pia/pia_features.csv

Lee el snapshot de HASE y sintetiza features para el motor de decisión PIA."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.loader import PROJECT_ROOT, get_config, get_path
from config.metadata import write_metadata

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pia_utils import (  # noqa: E402
    build_pia_dataset,
    load_snapshot_dataframe,
)

DEFAULT_SNAPSHOT_PATH = get_path('data', 'processed', 'hase', 'snapshot')
DEFAULT_OUTPUT_PATH = get_path('data', 'processed', 'pia', 'features')
DEFAULT_TARGET_PAYMENT = float(get_config('pia', 'target_payment', default=18000))


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build synthetic PIA feature dataset")
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=DEFAULT_SNAPSHOT_PATH,
        help="Ruta al snapshot de features por placa",
    )
    parser.add_argument(
        "--target-payment",
        type=float,
        default=DEFAULT_TARGET_PAYMENT,
        help="Pago mensual objetivo cubierto vía GNV (MXN)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Archivo destino con el dataset sintético",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla para variaciones reproducibles",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    snapshot_path = Path(args.snapshot)
    snapshot_df = load_snapshot_dataframe(snapshot_path)
    pia_df = build_pia_dataset(
        snapshot_df,
        target_payment=args.target_payment,
        seed=args.seed,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pia_df.to_csv(output_path, index=False)

    print(f"PIA synthetic dataset saved to {_rel(output_path)} (rows={len(pia_df)})")
    write_metadata(
        output_path,
        script=__file__,
        inputs=[snapshot_path],
        extra={"rows": int(len(pia_df)), "columns": int(len(pia_df.columns))}
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
