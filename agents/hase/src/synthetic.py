"""Utilities to generate synthetic HASE training rows for scenario coverage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SyntheticScenario:
    """Configuration for a synthetic scenario."""

    name: str
    count: int
    label_flag: int
    label_reason: str


NUMERIC_COLS = {
    "coverage_ratio_7d",
    "coverage_ratio_14d",
    "coverage_ratio_30d",
    "downtime_days_7d",
    "downtime_days_14d",
    "downtime_days_30d",
    "downtime_hours",
    "downtime_hours_7d",
    "downtime_hours_14d",
    "downtime_hours_30d",
    "litros_diarios",
    "litros_7d",
    "litros_14d",
    "litros_30d",
    "recaudo_diario",
    "recaudo_7d",
    "recaudo_14d",
    "recaudo_30d",
    "credito_diario",
    "credito_7d",
    "credito_14d",
    "credito_30d",
    "tickets_diarios",
    "tickets_7d",
    "tickets_14d",
    "tickets_30d",
    "activity_drop_pct",
    "recencia_dias",
    "protections_applied_last_12m",
    "days_since_last_protection",
}


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure required columns exist on dataframe (filled with defaults)."""

    for col in NUMERIC_COLS:
        if col not in df.columns:
            df[col] = 0.0
    for col in ("label_date", "label_reason", "plaza_limpia"):
        if col not in df.columns:
            df[col] = ""
    if "default_flag" not in df.columns:
        df["default_flag"] = 0
    return df


def _clip_positive(df: pd.DataFrame, minimum: float = 0.0) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=["number"]).columns
    df.loc[:, numeric_cols] = df.loc[:, numeric_cols].clip(lower=minimum)
    return df


def _apply_common_adjustments(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    df = df.copy()
    if "recencia_dias" in df.columns:
        df["recencia_dias"] = rng.integers(0, 30, size=len(df))
    if "protections_applied_last_12m" in df.columns:
        df["protections_applied_last_12m"] = rng.integers(0, 4, size=len(df))
    if "days_since_last_protection" in df.columns:
        df["days_since_last_protection"] = rng.integers(0, 180, size=len(df))
    return df


def _consumption_gap(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    df = df.copy()
    df["coverage_ratio_30d"] = rng.uniform(0.05, 0.25, size=len(df))
    df["coverage_ratio_14d"] = df["coverage_ratio_30d"] * rng.uniform(0.5, 0.8, size=len(df))
    df["coverage_ratio_7d"] = df["coverage_ratio_14d"] * rng.uniform(0.6, 0.9, size=len(df))
    for col in ["credito_diario", "credito_7d", "credito_14d", "credito_30d"]:
        df[col] = rng.uniform(0, 150, size=len(df))
    for col in ["recaudo_diario", "recaudo_7d", "recaudo_14d", "recaudo_30d", "litros_diarios", "litros_7d", "litros_14d", "litros_30d"]:
        df[col] = rng.uniform(0, 200, size=len(df))
    df["downtime_days_14d"] = rng.uniform(0, 1.5, size=len(df))
    df["downtime_days_30d"] = df["downtime_days_14d"] * rng.uniform(1.4, 2.2, size=len(df))
    df["downtime_days_7d"] = rng.uniform(0, 0.5, size=len(df))
    df["downtime_hours"] = rng.uniform(10, 30, size=len(df))
    df["downtime_hours_7d"] = df["downtime_days_7d"] * rng.uniform(18, 30, size=len(df))
    df["downtime_hours_14d"] = df["downtime_days_14d"] * rng.uniform(18, 30, size=len(df))
    df["downtime_hours_30d"] = df["downtime_days_30d"] * rng.uniform(18, 30, size=len(df))
    df["activity_drop_pct"] = rng.uniform(0.5, 0.9, size=len(df))
    return df


def _downtime_spike(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    df = df.copy()
    df["coverage_ratio_30d"] = rng.uniform(0.2, 0.6, size=len(df))
    df["coverage_ratio_14d"] = df["coverage_ratio_30d"] * rng.uniform(0.6, 0.9, size=len(df))
    df["downtime_days_14d"] = rng.uniform(6, 12, size=len(df))
    df["downtime_days_30d"] = rng.uniform(14, 25, size=len(df))
    df["downtime_days_7d"] = rng.uniform(3, 6, size=len(df))
    df["downtime_hours"] = rng.uniform(120, 240, size=len(df))
    df["downtime_hours_7d"] = df["downtime_days_7d"] * rng.uniform(18, 24, size=len(df))
    df["downtime_hours_14d"] = df["downtime_days_14d"] * rng.uniform(18, 24, size=len(df))
    df["downtime_hours_30d"] = df["downtime_days_30d"] * rng.uniform(18, 24, size=len(df))
    df["litros_30d"] = rng.uniform(0, 150, size=len(df))
    df["recaudo_30d"] = rng.uniform(0, 500, size=len(df))
    df["activity_drop_pct"] = rng.uniform(0.6, 1.0, size=len(df))
    return df


def _fault_alert(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    df = df.copy()
    df["coverage_ratio_30d"] = rng.uniform(0.3, 0.8, size=len(df))
    df["coverage_ratio_14d"] = df["coverage_ratio_30d"] * rng.uniform(0.7, 0.95, size=len(df))
    df["downtime_days_14d"] = rng.uniform(4, 8, size=len(df))
    df["downtime_days_30d"] = rng.uniform(10, 18, size=len(df))
    df["downtime_days_7d"] = rng.uniform(1, 3, size=len(df))
    df["downtime_hours"] = rng.uniform(80, 200, size=len(df))
    df["downtime_hours_7d"] = df["downtime_days_7d"] * rng.uniform(18, 30, size=len(df))
    df["downtime_hours_14d"] = df["downtime_days_14d"] * rng.uniform(18, 30, size=len(df))
    df["downtime_hours_30d"] = df["downtime_days_30d"] * rng.uniform(18, 30, size=len(df))
    df["activity_drop_pct"] = rng.uniform(0.4, 0.8, size=len(df))
    df["litros_30d"] = rng.uniform(50, 300, size=len(df))
    df["recaudo_30d"] = rng.uniform(500, 1500, size=len(df))
    return df


def _stable_high(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    df = df.copy()
    df["coverage_ratio_30d"] = rng.uniform(1.2, 2.2, size=len(df))
    df["coverage_ratio_14d"] = df["coverage_ratio_30d"] * rng.uniform(0.9, 1.1, size=len(df))
    df["coverage_ratio_7d"] = df["coverage_ratio_14d"] * rng.uniform(0.9, 1.1, size=len(df))
    for col in ["downtime_days_7d", "downtime_days_14d", "downtime_days_30d"]:
        df[col] = rng.uniform(0, 0.4, size=len(df))
    for col in ["downtime_hours", "downtime_hours_7d", "downtime_hours_14d", "downtime_hours_30d"]:
        df[col] = rng.uniform(0, 6, size=len(df))
    df["litros_diarios"] = rng.uniform(400, 800, size=len(df))
    df["credito_diario"] = rng.uniform(400, 800, size=len(df))
    df["recaudo_diario"] = rng.uniform(400, 800, size=len(df))
    for target, source in [("litros_7d", "litros_diarios"), ("litros_14d", "litros_diarios"), ("litros_30d", "litros_diarios"),
                           ("recaudo_7d", "recaudo_diario"), ("recaudo_14d", "recaudo_diario"), ("recaudo_30d", "recaudo_diario"),
                           ("credito_7d", "credito_diario"), ("credito_14d", "credito_diario"), ("credito_30d", "credito_diario")]:
        df[target] = df[source] * rng.uniform(5, 25, size=len(df))
    for col in ["tickets_diarios", "tickets_7d", "tickets_14d", "tickets_30d"]:
        df[col] = rng.uniform(5, 20, size=len(df))
    df["activity_drop_pct"] = rng.uniform(0.0, 0.2, size=len(df))
    df["recencia_dias"] = rng.integers(0, 5, size=len(df))
    return df


def _recovery(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    df = df.copy()
    df["coverage_ratio_30d"] = rng.uniform(0.8, 1.2, size=len(df))
    df["coverage_ratio_14d"] = df["coverage_ratio_30d"] * rng.uniform(0.95, 1.05, size=len(df))
    df["coverage_ratio_7d"] = df["coverage_ratio_14d"] * rng.uniform(0.95, 1.05, size=len(df))
    df["downtime_days_14d"] = rng.uniform(0, 2, size=len(df))
    df["downtime_days_30d"] = rng.uniform(1, 4, size=len(df))
    df["downtime_days_7d"] = rng.uniform(0, 1, size=len(df))
    df["downtime_hours"] = rng.uniform(10, 60, size=len(df))
    df["activity_drop_pct"] = rng.uniform(0.1, 0.3, size=len(df))
    df["litros_30d"] = rng.uniform(300, 800, size=len(df))
    df["recaudo_30d"] = rng.uniform(1000, 4000, size=len(df))
    return df


SCENARIO_BUILDERS: Dict[str, Callable[[pd.DataFrame, np.random.Generator], pd.DataFrame]] = {
    "consumption_gap": _consumption_gap,
    "downtime_spike": _downtime_spike,
    "fault_alert": _fault_alert,
    "stable_high": _stable_high,
    "recovery": _recovery,
}


def generate_synthetic_rows(
    base: pd.DataFrame,
    scenarios: Dict[str, SyntheticScenario],
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Generate synthetic rows based on scenario configuration."""

    rng = rng or np.random.default_rng()
    base = _ensure_columns(base.copy())
    synthetic_frames = []

    for key, scenario in scenarios.items():
        if scenario.name not in SCENARIO_BUILDERS:
            raise ValueError(f"Unknown scenario '{scenario.name}'")
        if scenario.count <= 0:
            continue
        sampler = base.sample(n=scenario.count, replace=True, random_state=rng.integers(0, 10**9))
        sampler = sampler.reset_index(drop=True)
        sampler = _apply_common_adjustments(sampler, rng)
        sampler = SCENARIO_BUILDERS[scenario.name](sampler, rng)
        sampler = _clip_positive(sampler)
        sampler["default_flag"] = scenario.label_flag
        sampler["label_reason"] = scenario.label_reason
        sampler["label_date"] = pd.Timestamp.utcnow().normalize().date()
        synthetic_frames.append(sampler)

    if not synthetic_frames:
        return pd.DataFrame(columns=base.columns)
    synthetic = pd.concat(synthetic_frames, ignore_index=True)
    return synthetic


__all__ = ["SyntheticScenario", "generate_synthetic_rows"]
