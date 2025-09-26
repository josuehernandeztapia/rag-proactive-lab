#!/usr/bin/env python3
"""Generate daily risk report for plates using coverage and telemetría signals."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd

DEFAULT_COLUMNS = [
    "plaza_limpia",
    "placa",
    "coverage_ratio_14d",
    "coverage_ratio_30d",
    "downtime_days_14d",
    "downtime_days_30d",
    "protections_applied_last_12m",
]


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate risk report from feature store")
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("data/hase/consumos_features_daily.csv.gz"),
        help="Path to daily features file",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("notebooks/hase/placas_en_riesgo_gnv.csv"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("notebooks/hase/placas_en_riesgo_gnv.md"),
    )
    parser.add_argument("--cov14-threshold", type=float, default=0.7)
    parser.add_argument("--cov30-threshold", type=float, default=0.8)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    df = pd.read_csv(args.features)
    for column in DEFAULT_COLUMNS:
        if column not in df.columns:
            raise ValueError(f"Missing column '{column}' in features file")

    mask = (df["coverage_ratio_14d"].fillna(1) < args.cov14_threshold) & \
           (df["coverage_ratio_30d"].fillna(1) < args.cov30_threshold)
    risk_df = df.loc[mask, DEFAULT_COLUMNS].copy()

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    risk_df.to_csv(args.output_csv, index=False)

    summary = risk_df.groupby("plaza_limpia").size().sort_values(ascending=False)
    top10 = risk_df.head(10)

    lines = [
        "# Placas en riesgo (coverage <70% 14d & <80% 30d)",
        "",
        f"Fuente: {args.features}",
        "",
        "## Totales por plaza",
        summary.to_string(),
        "",
        "## Top 10 placas críticas",
        top10.to_string(index=False),
    ]

    args.output_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Riesgo generado: {args.output_csv} ({len(risk_df)} placas)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
