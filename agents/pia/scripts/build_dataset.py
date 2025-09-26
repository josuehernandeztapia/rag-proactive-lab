#!/usr/bin/env python3
"""Generate a synthetic PIA dataset using HASE feature store."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pia_utils import (  # noqa: E402
    DEFAULT_TARGET_PAYMENT,
    build_pia_dataset,
    load_snapshot_dataframe,
)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build synthetic PIA feature dataset")
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=Path("data/hase/consumos_snapshot_latest.csv.gz"),
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
        default=Path("data/pia/pia_features.csv"),
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
    snapshot_df = load_snapshot_dataframe(Path(args.snapshot))
    pia_df = build_pia_dataset(
        snapshot_df,
        target_payment=args.target_payment,
        seed=args.seed,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pia_df.to_csv(args.output, index=False)

    print(f"PIA synthetic dataset saved to {args.output} (rows={len(pia_df)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
