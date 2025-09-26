"""Helpers to generate synthetic PIA datasets and scenario summaries."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ScenarioConfig:
    prob: float
    arrears_low: float
    arrears_high: float
    intent: str = ""


SCENARIOS: Dict[str, ScenarioConfig] = {
    "late_payment": ScenarioConfig(prob=0.22, arrears_low=800, arrears_high=3500),
    "prepayment": ScenarioConfig(prob=0.14, arrears_low=-1200, arrears_high=-200),
    "balance_inquiry": ScenarioConfig(prob=0.12, arrears_low=0, arrears_high=150, intent="balance_inquiry"),
    "advance_payment": ScenarioConfig(prob=0.08, arrears_low=0, arrears_high=200, intent="advance_payment"),
    "downtime_spike": ScenarioConfig(prob=0.18, arrears_low=200, arrears_high=2000),
    "consumption_gap": ScenarioConfig(prob=0.14, arrears_low=600, arrears_high=2600),
    "fault_alert": ScenarioConfig(prob=0.12, arrears_low=200, arrears_high=1800),
}


def _ensure_intent_column(df: pd.DataFrame) -> pd.DataFrame:
    intent_col = df["intent"] if "intent" in df.columns else pd.Series("", index=df.index)
    df["intent"] = intent_col.fillna("")
    return df


def _ensure_column(df: pd.DataFrame, name: str, values: np.ndarray) -> pd.DataFrame:
    filler = pd.Series(values, index=df.index)
    if name in df.columns:
        df[name] = df[name].fillna(filler)
    else:
        df[name] = filler
    return df


def augment_dataframe(df: pd.DataFrame, rng: np.random.Generator | None = None) -> pd.DataFrame:
    rng = rng or np.random.default_rng()
    df = df.copy()
    df["scenario"] = "base"
    df = _ensure_intent_column(df)

    n_rows = len(df)
    df = _ensure_column(df, "distance_km_30d", rng.uniform(600, 1600, size=n_rows))
    df = _ensure_column(df, "engine_hours_30d", rng.uniform(140, 260, size=n_rows))
    df = _ensure_column(df, "fault_events_30d", rng.poisson(0.6, size=n_rows))
    df = _ensure_column(df, "gps_operational_ratio", rng.uniform(0.85, 0.99, size=n_rows))
    baseline_numeric = [
        "avg_daily_litros",
        "avg_daily_recaudo",
        "litros_30d",
        "litros_90d",
        "litros_180d",
        "litros_365d",
        "recaudo_30d",
        "recaudo_90d",
        "recaudo_180d",
        "recaudo_365d",
        "avg_30d_litros",
        "avg_30d_recaudo",
        "max_30d_litros",
        "max_30d_recaudo",
    ]
    for col in baseline_numeric:
        df = _ensure_column(df, col, np.zeros(n_rows))
    for col in ["records_30d", "records_90d", "records_180d", "records_365d", "records_total"]:
        if col not in df.columns:
            df[col] = 0
    for col in ["first_record_date", "last_record_date", "plaza"]:
        if col not in df.columns:
            df[col] = ""
    if "plaza_limpia" not in df.columns:
        base_plaza = df["plaza"] if "plaza" in df.columns else pd.Series("UNKNOWN", index=df.index)
        df["plaza_limpia"] = base_plaza.fillna("UNKNOWN")
    if "hase_consumption_gap_flag" not in df.columns:
        df["hase_consumption_gap_flag"] = 0
    if "hase_fault_alert_flag" not in df.columns:
        df["hase_fault_alert_flag"] = 0
    if "hase_downtime_alert_flag" not in df.columns:
        df["hase_downtime_alert_flag"] = 0
    if "hase_telemetry_ok_flag" not in df.columns:
        df["hase_telemetry_ok_flag"] = 0

    variations = [df]
    for label, cfg in SCENARIOS.items():
        sample_size = max(1, int(n_rows * cfg.prob))
        indices = rng.choice(n_rows, size=sample_size, replace=False)
        variant = df.iloc[indices].copy()
        variant["scenario"] = label
        variant["arrears_amount"] = rng.uniform(cfg.arrears_low, cfg.arrears_high, size=sample_size)
        if cfg.intent:
            variant["intent"] = cfg.intent

        if label == "prepayment":
            variant["bank_transfer"] = variant.get("bank_transfer", 0) + rng.uniform(4000, 6000, size=sample_size)
            variant["gnv_credit_30d"] = variant.get("gnv_credit_30d", 0) + rng.uniform(5000, 7000, size=sample_size)
        elif label == "downtime_spike":
            variant["downtime_hours_30d"] = variant.get("downtime_hours_30d", 0) + rng.uniform(120, 220, size=sample_size)
            variant["coverage_ratio_14d"] = (variant.get("coverage_ratio_14d", 0.6) - rng.uniform(0.15, 0.3, size=sample_size)).clip(0, 1)
            variant["coverage_ratio_30d"] = (variant.get("coverage_ratio_30d", 0.6) - rng.uniform(0.1, 0.2, size=sample_size)).clip(0, 1)
            variant["hase_downtime_alert_flag"] = 1
        elif label == "consumption_gap":
            variant["gnv_credit_30d"] = rng.uniform(0, 300, size=sample_size)
            variant["bank_transfer"] = rng.uniform(0, 600, size=sample_size)
            variant["distance_km_30d"] = rng.uniform(900, 1600, size=sample_size)
            variant["engine_hours_30d"] = rng.uniform(200, 320, size=sample_size)
            variant["gps_operational_ratio"] = rng.uniform(0.9, 0.99, size=sample_size)
            variant["hase_consumption_gap_flag"] = 1
            variant["hase_telemetry_ok_flag"] = 1
        elif label == "fault_alert":
            variant["fault_events_30d"] = rng.poisson(2.5, size=sample_size) + 1
            variant["downtime_hours_30d"] = variant.get("downtime_hours_30d", 0) + rng.uniform(40, 120, size=sample_size)
            variant["coverage_ratio_14d"] = (variant.get("coverage_ratio_14d", 0.6) - rng.uniform(0.05, 0.15, size=sample_size)).clip(0, 1)
            variant["hase_fault_alert_flag"] = 1

        if label in {"late_payment", "downtime_spike"}:
            variant["risk_score"] = np.clip(variant.get("risk_score", 0.7) + rng.uniform(0.1, 0.2, size=sample_size), 0, 1)
        elif label == "prepayment":
            variant["risk_score"] = np.clip(variant.get("risk_score", 0.7) - rng.uniform(0.1, 0.2, size=sample_size), 0, 1)
        elif label == "consumption_gap":
            variant["risk_score"] = np.clip(variant.get("risk_score", 0.7) + rng.uniform(0.05, 0.15, size=sample_size), 0, 1)
        elif label == "fault_alert":
            variant["risk_score"] = np.clip(variant.get("risk_score", 0.7) + rng.uniform(0.08, 0.18, size=sample_size), 0, 1)

        variations.append(variant)

    combined = pd.concat(variations, ignore_index=True).drop_duplicates()
    return combined


def summarize_scenarios(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = _ensure_intent_column(df)
    if "scenario" not in df.columns:
        df["scenario"] = "base"
    else:
        df["scenario"] = df["scenario"].fillna("base")
    n_rows = len(df)
    df = _ensure_column(df, "distance_km_30d", np.zeros(n_rows))
    df = _ensure_column(df, "engine_hours_30d", np.zeros(n_rows))
    df = _ensure_column(df, "gnv_credit_30d", np.zeros(n_rows))
    df = _ensure_column(df, "hase_consumption_gap_flag", np.zeros(n_rows))
    df = _ensure_column(df, "hase_fault_alert_flag", np.zeros(n_rows))
    if "plaza_limpia" not in df.columns:
        base_plaza = df["plaza"] if "plaza" in df.columns else pd.Series("UNKNOWN", index=df.index)
        df["plaza_limpia"] = base_plaza.fillna("UNKNOWN")
    df["arrears_bucket"] = pd.cut(
        df["arrears_amount"], bins=[-5000, -100, 0, 500, 2000, 5000], include_lowest=True
    )
    summary = (
        df.groupby(["plaza_limpia", "scenario", "intent", "arrears_bucket"], dropna=False, observed=True)
        .agg(
            registros=("placa", "count"),
            riesgo_medio=("risk_score", "mean"),
            cobertura14_media=("coverage_ratio_14d", "mean"),
            downtime30_medio=("downtime_hours_30d", "mean"),
            distancia_media=("distance_km_30d", "mean"),
            horas_motor_medias=("engine_hours_30d", "mean"),
            consumo_gnv_medio=("gnv_credit_30d", "mean"),
            consumo_gap_ratio=("hase_consumption_gap_flag", "mean"),
            fault_ratio=("hase_fault_alert_flag", "mean"),
        )
        .reset_index()
    )
    summary = summary[summary["registros"] > 0].reset_index(drop=True)
    return summary


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

# TODO: conectar con telemetría/pagos reales.
