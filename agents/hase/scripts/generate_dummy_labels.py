#!/usr/bin/env python3
"""Generate synthetic default labels using coverage & telemetría signals."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

RANDOM_SEED = 42


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate dummy labels for HASE training")
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("data/hase/consumos_snapshot_latest.csv.gz"),
        help="Features file (snapshot) used as base",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/hase/dummy_labels.csv"),
        help="Destination CSV with columns placa, label_date, default_flag, reason",
    )
    parser.add_argument(
        "--coverage-low",
        type=float,
        default=0.5,
        help="Threshold coverage_ratio_30d below which we consider risk",
    )
    parser.add_argument(
        "--downtime-high",
        type=float,
        default=2.0,
        help="Threshold downtime_days_14d above which we consider unit inactive",
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=0.1,
        help="Fraction of random positives to keep model flexible",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    df = pd.read_csv(args.features)
    if "coverage_ratio_30d" not in df or "downtime_days_14d" not in df:
        raise ValueError("Features file must contain coverage_ratio_30d and downtime_days_14d")

    df["label_date"] = pd.to_datetime(df["fecha_dia"], errors="coerce")

    # Risk heuristics
    low_coverage = df["coverage_ratio_30d"].fillna(0) < args.coverage_low
    downtime_high = df["downtime_days_14d"].fillna(0) >= args.downtime_high
    coverage_drop = df["coverage_ratio_14d"].fillna(0) < args.coverage_low * 0.8

    reasons = []
    default_flag = np.zeros(len(df), dtype=int)

    for idx, row in df.iterrows():
        reason = []
        if low_coverage.iloc[idx] and not downtime_high.iloc[idx]:  # cobertura baja, unidad activa -> posible gasolina
            default_flag[idx] = 1
            reason.append("low_coverage_active")
        elif low_coverage.iloc[idx] and downtime_high.iloc[idx]:  # cobertura baja + inactiva -> shock ingresos
            default_flag[idx] = 1
            reason.append("low_coverage_inactive")
        elif coverage_drop.iloc[idx] and row.get("protections_applied_last_12m", 0) > 0:
            default_flag[idx] = 1
            reason.append("coverage_drop_with_protection")
        else:
            default_flag[idx] = 0
            reason.append("stable")
        reasons.append("|".join(reason))

    rng = np.random.default_rng(RANDOM_SEED)
    noise_mask = rng.random(len(df)) < args.noise
    default_flag = np.where((default_flag == 0) & noise_mask, 1, default_flag)
    reasons = [r + "|noise" if noise_mask[i] and default_flag[i] == 1 else r for i, r in enumerate(reasons)]

    output_df = pd.DataFrame({
        "placa": df["placa"],
        "label_date": df["label_date"].dt.date,
        "default_flag": default_flag,
        "reason": reasons,
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(args.output, index=False)
    print(f"Dummy labels saved to {args.output} (positives={output_df['default_flag'].sum()})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
