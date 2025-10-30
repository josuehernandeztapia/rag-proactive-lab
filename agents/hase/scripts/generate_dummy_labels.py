#!/usr/bin/env python3
"""Generate synthetic default labels using coverage & telemetría signals.

Usage:
    python agents/hase/scripts/generate_dummy_labels.py
    python agents/hase/scripts/generate_dummy_labels.py \
        --features data/processed/hase/consumos_snapshot_latest.csv.gz \
        --output data/processed/hase/dummy_labels.csv

Genera `default_flag` y `reason` heurísticos para bootstrapping del modelo HASE."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.loader import PROJECT_ROOT, get_config, get_path
from config.metadata import write_metadata

RANDOM_SEED = 42

SAFETY_CONFIG = get_config('pia', 'safety', default={}) or {}
SEATBELT_THRESHOLD = float(SAFETY_CONFIG.get('seatbelt_threshold', 0.30))
HIGH_SPEED_THRESHOLD = float(SAFETY_CONFIG.get('high_speed_threshold', 0.34))
IDLE_RATIO_THRESHOLD = float(SAFETY_CONFIG.get('idle_ratio_threshold', 0.60))
AFTER_HOURS_THRESHOLD = float(SAFETY_CONFIG.get('after_hours_ratio_threshold', 0.35))

DEFAULT_FEATURES_PATH = get_path('data', 'processed', 'hase', 'snapshot')
DEFAULT_OUTPUT_PATH = get_path('data', 'processed', 'hase', 'dummy_labels')


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate dummy labels for HASE training")
    parser.add_argument(
        "--features",
        type=Path,
        default=DEFAULT_FEATURES_PATH,
        help="Features file (snapshot) used as base",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
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
    features_path = Path(args.features)
    df = pd.read_csv(features_path)
    if "coverage_ratio_30d" not in df or "downtime_days_14d" not in df:
        raise ValueError("Features file must contain coverage_ratio_30d and downtime_days_14d")

    df["label_date"] = pd.to_datetime(df["fecha_dia"], errors="coerce")

    # Risk heuristics
    low_coverage = df["coverage_ratio_30d"].fillna(0) < args.coverage_low
    downtime_high = df["downtime_days_14d"].fillna(0) >= args.downtime_high
    coverage_drop = df["coverage_ratio_14d"].fillna(0) < args.coverage_low * 0.8

    seatbelt_rate = df.get("seatbelt_off_rate_30d", 0).fillna(0)
    high_speed_ratio = df.get("high_speed_ratio_30d", 0).fillna(0)
    idle_ratio = df.get("idle_hours_ratio_30d", 0).fillna(0)
    activity_drop = df.get("activity_drop_pct", 0).fillna(0)
    distance_30d = df.get("distance_km_30d", 0).fillna(0)
    after_hours_distance = df.get("after_hours_distance_km_30d", 0).fillna(0)
    after_hours_ratio = np.where(
        distance_30d > 0,
        after_hours_distance / np.maximum(distance_30d, 1e-6),
        0,
    )

    seatbelt_flag = pd.Series(seatbelt_rate >= SEATBELT_THRESHOLD, index=df.index)
    speed_flag = pd.Series(high_speed_ratio >= HIGH_SPEED_THRESHOLD, index=df.index)
    idle_flag = pd.Series(idle_ratio >= IDLE_RATIO_THRESHOLD, index=df.index)
    after_hours_flag = pd.Series(after_hours_ratio >= AFTER_HOURS_THRESHOLD, index=df.index)
    activity_flag = pd.Series(activity_drop > 0.6, index=df.index)

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

        if seatbelt_flag.iloc[idx]:
            default_flag[idx] = 1
            reason.append("seatbelt_anomaly")
        if speed_flag.iloc[idx]:
            default_flag[idx] = 1
            reason.append("speeding_pattern")
        if idle_flag.iloc[idx]:
            default_flag[idx] = 1
            reason.append("idle_overload")
        if after_hours_flag.iloc[idx]:
            default_flag[idx] = 1
            reason.append("after_hours_heavy")
        if activity_flag.iloc[idx]:
            default_flag[idx] = 1
            reason.append("activity_collapse")
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
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    positives = int(output_df["default_flag"].sum())
    print(f"Dummy labels saved to {_rel(output_path)} (positives={positives})")
    write_metadata(
        output_path,
        script=__file__,
        inputs=[features_path],
        extra={"rows": int(len(output_df)), "positives": positives}
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
