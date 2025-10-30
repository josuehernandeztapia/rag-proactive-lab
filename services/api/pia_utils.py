"""Utilities for PIA synthetic dataset generation and serving."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from config.loader import PROJECT_ROOT, get_config, get_path

DEFAULT_TARGET_PAYMENT = float(get_config('pia', 'target_payment', default=11000))
SAFETY_CONFIG = get_config('pia', 'safety', default={}) or {}
SAFETY_SEATBELT_THRESHOLD = float(SAFETY_CONFIG.get('seatbelt_threshold', 0.30))
SAFETY_SPEED_THRESHOLD = float(SAFETY_CONFIG.get('high_speed_threshold', 0.34))
IDLE_RATIO_THRESHOLD = float(SAFETY_CONFIG.get('idle_ratio_threshold', 0.60))
AFTER_HOURS_RATIO_THRESHOLD = float(SAFETY_CONFIG.get('after_hours_ratio_threshold', 0.35))
TELEMETRY_HEALTH_THRESHOLD = float(SAFETY_CONFIG.get('telemetry_health_threshold', 0.40))


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _candidate_snapshot_paths() -> list[Path]:
    cwd = Path.cwd()
    candidates = [
        get_path('data', 'processed', 'hase', 'snapshot'),
        Path('conductores/data/processed/hase/consumos_snapshot_latest.csv.gz'),
    ]
    seen: set[Path] = set()
    ordered: list[Path] = []
    for cand in candidates:
        c = cand if cand.is_absolute() else (cwd / cand)
        if c not in seen and c.exists():
            ordered.append(c)
            seen.add(c)
    return ordered


def _candidate_pia_dataset_paths() -> list[Path]:
    cwd = Path.cwd()
    candidates = [
        get_path('data', 'processed', 'pia', 'features'),
        Path('conductores/data/processed/pia/pia_features.csv'),
    ]
    seen: set[Path] = set()
    ordered: list[Path] = []
    for cand in candidates:
        c = cand if cand.is_absolute() else (cwd / cand)
        if c not in seen and c.exists():
            ordered.append(c)
            seen.add(c)
    return ordered


def load_snapshot_dataframe(path: Path | None = None) -> pd.DataFrame:
    """Load the latest snapshot dataframe from known locations."""
    if path is not None:
        return pd.read_csv(path)
    for candidate in _candidate_snapshot_paths():
        try:
            return pd.read_csv(candidate)
        except Exception:
            continue
    attempts = ", ".join(_rel(p) for p in _candidate_snapshot_paths())
    raise FileNotFoundError(f"No se encontró el snapshot de consumos para PIA; intenté: {attempts}")


def _ensure_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")
    return df


def build_pia_dataset(
    snapshot_df: pd.DataFrame,
    target_payment: float = DEFAULT_TARGET_PAYMENT,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic PIA dataset from the latest snapshot."""

    snapshot_df = snapshot_df.copy()
    rng = np.random.default_rng(seed)

    _ensure_columns(
        snapshot_df,
        [
            "plaza_limpia",
            "placa",
            "fecha_dia",
            "coverage_ratio_14d",
            "coverage_ratio_30d",
            "credito_30d",
            "downtime_hours_30d",
            "activity_drop_pct",
            "average_speed_kph_30d",
            "trip_count_30d",
            "distance_km_30d",
            "seatbelt_off_rate_30d",
            "high_speed_ratio_30d",
            "idle_hours_ratio_30d",
        ],
    )

    snapshot_df["protections_applied_last_12m"] = (
        snapshot_df.get("protections_applied_last_12m", 0).fillna(0).astype(int)
    )
    snapshot_df["last_protection_at"] = pd.to_datetime(
        snapshot_df.get("last_protection_at"), errors="coerce"
    )

    snapshot_df["downtime_hours_30d"] = snapshot_df.get(
        "downtime_hours_30d", snapshot_df.get("downtime_hours", 0)
    ).fillna(0)
    snapshot_df["activity_drop_pct"] = (
        snapshot_df.get("activity_drop_pct", 0).fillna(0).clip(0, 1)
    )

    snapshot_df["engine_hours_30d"] = snapshot_df.get(
        "engine_hours_30d", snapshot_df.get("engine_hours", 0)
    ).fillna(0)
    snapshot_df["driving_hours_30d"] = snapshot_df.get(
        "driving_hours_30d", snapshot_df.get("driving_hours", 0)
    ).fillna(0)
    snapshot_df["idling_hours_30d"] = snapshot_df.get(
        "idling_hours_30d", snapshot_df.get("idling_hours", 0)
    ).fillna(0)
    snapshot_df["after_hours_distance_km_30d"] = snapshot_df.get(
        "after_hours_distance_km_30d", snapshot_df.get("after_hours_distance_km", 0)
    ).fillna(0)
    snapshot_df["after_hours_driving_hours_30d"] = snapshot_df.get(
        "after_hours_driving_hours_30d", snapshot_df.get("after_hours_driving_hours", 0)
    ).fillna(0)
    snapshot_df["after_hours_stop_hours_30d"] = snapshot_df.get(
        "after_hours_stop_hours_30d", snapshot_df.get("after_hours_stop_hours", 0)
    ).fillna(0)
    snapshot_df["work_distance_km_30d"] = snapshot_df.get(
        "work_distance_km_30d", snapshot_df.get("work_distance_km", 0)
    ).fillna(0)
    snapshot_df["work_driving_hours_30d"] = snapshot_df.get(
        "work_driving_hours_30d", snapshot_df.get("work_driving_hours", 0)
    ).fillna(0)
    snapshot_df["work_stop_hours_30d"] = snapshot_df.get(
        "work_stop_hours_30d", snapshot_df.get("work_stop_hours", 0)
    ).fillna(0)

    snapshot_df["coverage_ratio_30d"] = snapshot_df["coverage_ratio_30d"].fillna(0)
    snapshot_df["coverage_ratio_14d"] = snapshot_df["coverage_ratio_14d"].fillna(
        snapshot_df["coverage_ratio_30d"]
    )

    snapshot_df["gnv_credit_30d"] = snapshot_df["credito_30d"].fillna(0)
    snapshot_df["expected_payment"] = float(target_payment)
    snapshot_df["arrears_amount"] = (
        snapshot_df["expected_payment"] - snapshot_df["gnv_credit_30d"]
    ).clip(lower=0)

    proportion = rng.uniform(0.3, 0.9, size=len(snapshot_df))
    snapshot_df["bank_transfer"] = (
        (snapshot_df["arrears_amount"] * proportion).round(2)
    )
    snapshot_df["exposure_after_transfer"] = (
        snapshot_df["arrears_amount"] - snapshot_df["bank_transfer"]
    ).clip(lower=0)

    coverage_gap = (1 - snapshot_df["coverage_ratio_30d"].clip(0, 1)).clip(lower=0)
    downtime_norm = (snapshot_df["downtime_hours_30d"] / 72).clip(0, 1)
    activity_norm = snapshot_df["activity_drop_pct"].clip(0, 1)
    arrears_ratio = np.where(
        snapshot_df["expected_payment"] > 0,
        snapshot_df["arrears_amount"] / snapshot_df["expected_payment"],
        0,
    )
    arrears_ratio = np.clip(arrears_ratio, 0, 1)

    distance = snapshot_df["distance_km_30d"].replace(0, np.nan)
    after_hours_ratio = np.clip(
        snapshot_df["after_hours_distance_km_30d"] / distance,
        0,
        1,
    ).fillna(0)
    after_hours_pressure = np.clip(after_hours_ratio / max(AFTER_HOURS_RATIO_THRESHOLD, 1e-6), 0, 1)

    seatbelt_pressure = np.clip(
        snapshot_df["seatbelt_off_rate_30d"] / max(SAFETY_SEATBELT_THRESHOLD, 1e-6),
        0,
        1,
    )
    speed_pressure = np.clip(
        snapshot_df["high_speed_ratio_30d"] / max(SAFETY_SPEED_THRESHOLD, 1e-6),
        0,
        1,
    )
    snapshot_df["safety_score"] = (0.6 * seatbelt_pressure + 0.4 * speed_pressure).clip(0, 1).round(3)
    snapshot_df["safety_alert"] = (
        (seatbelt_pressure >= 1.0)
        | (speed_pressure >= 1.0)
    ).astype(int)

    idle_norm = np.clip(snapshot_df["idle_hours_ratio_30d"] / max(IDLE_RATIO_THRESHOLD, 1e-6), 0, 1)
    downtime_pressure = np.clip(snapshot_df["downtime_hours_30d"] / 120, 0, 1)
    snapshot_df["telemetry_health_score"] = (
        1
        - (
            0.35 * downtime_pressure
            + 0.25 * idle_norm
            + 0.20 * snapshot_df["activity_drop_pct"].clip(0, 1)
            + 0.20 * snapshot_df["safety_score"].clip(0, 1)
        )
    ).clip(0, 1)
    snapshot_df["telemetry_alert"] = (
        snapshot_df["telemetry_health_score"] <= TELEMETRY_HEALTH_THRESHOLD
    ).astype(int)

    snapshot_df["risk_score"] = (
        0.30 * coverage_gap
        + 0.20 * downtime_norm
        + 0.12 * activity_norm
        + 0.10 * idle_norm
        + 0.08 * speed_pressure
        + 0.07 * seatbelt_pressure
        + 0.05 * after_hours_pressure
        + 0.08 * arrears_ratio
    ).clip(0, 1).round(3)

    snapshot_df["needs_protection"] = (
        (snapshot_df["risk_score"] > 0.45)
        | (snapshot_df["coverage_ratio_14d"] < 0.65)
        | (snapshot_df["exposure_after_transfer"] > 2500)
        | (snapshot_df["safety_alert"] == 1)
        | (idle_norm >= 1)
    ).astype(int)

    restructure_full = (snapshot_df["downtime_hours_30d"] > 72) & (
        snapshot_df["coverage_ratio_30d"] < 0.5
    )
    restructure_light = snapshot_df["coverage_ratio_30d"] < 0.6
    idle_review = idle_norm >= 1
    monitor_ready = (snapshot_df["needs_protection"] == 0) & (
        snapshot_df["safety_alert"] == 0
    )

    conditions = [
        snapshot_df["safety_alert"] == 1,
        restructure_full,
        restructure_light,
        idle_review,
        monitor_ready,
    ]
    scenarios = [
        "safety-coaching",
        "restructure-full",
        "restructure-light",
        "idle-rebalance",
        "monitor",
    ]
    snapshot_df["suggested_scenario"] = np.select(
        conditions, scenarios, default="advisor-review"
    )

    snapshot_df["whatsapp_segment"] = np.select(
        [snapshot_df["safety_alert"] == 1, snapshot_df["needs_protection"] == 1],
        ["PIA_SEGUIMIENTO", "PIA_OPCIONES"],
        default="FOLLOW_UP",
    )

    snapshot_df["after_hours_ratio_30d"] = after_hours_ratio.round(3)
    snapshot_df["idle_pressure_30d"] = idle_norm.round(3)
    snapshot_df["high_speed_pressure_30d"] = speed_pressure.round(3)
    snapshot_df["seatbelt_pressure_30d"] = seatbelt_pressure.round(3)

    useful_cols = [
        "plaza_limpia",
        "placa",
        "fecha_dia",
        "coverage_ratio_14d",
        "coverage_ratio_30d",
        "downtime_hours_30d",
        "activity_drop_pct",
        "average_speed_kph_30d",
        "trip_count_30d",
        "distance_km_30d",
        "engine_hours_30d",
        "driving_hours_30d",
        "idling_hours_30d",
        "after_hours_distance_km_30d",
        "work_distance_km_30d",
        "after_hours_ratio_30d",
        "idle_pressure_30d",
        "high_speed_pressure_30d",
        "seatbelt_pressure_30d",
        "telemetry_health_score",
        "telemetry_alert",
        "safety_score",
        "safety_alert",
        "protections_applied_last_12m",
        "last_protection_at",
        "expected_payment",
        "gnv_credit_30d",
        "bank_transfer",
        "exposure_after_transfer",
        "arrears_amount",
        "risk_score",
        "needs_protection",
        "suggested_scenario",
        "whatsapp_segment",
    ]

    return snapshot_df[useful_cols]


@lru_cache(maxsize=1)
def load_pia_dataset(path: Path | None = None) -> pd.DataFrame:
    if path is not None:
        return pd.read_csv(path)
    for candidate in _candidate_pia_dataset_paths():
        try:
            return pd.read_csv(candidate)
        except Exception:
            continue
    raise FileNotFoundError("No se encontró el dataset PIA (data/processed/pia/pia_features.csv)")


def get_driver_record(placa: str) -> dict | None:
    df = load_pia_dataset()
    row = df.loc[df["placa"].astype(str) == str(placa)].head(1)
    if row.empty:
        return None
    rec = row.iloc[0].to_dict()
    if isinstance(rec.get("last_protection_at"), str):
        rec["last_protection_at"] = rec["last_protection_at"] or None
    return rec


@dataclass
class SimulationResult:
    placa: str
    risk_score: float
    needs_protection: int
    suggested_scenario: str
    whatsapp_segment: str


def simulate_from_payload(payload: dict, target_payment: float | None = None) -> SimulationResult:
    tp = target_payment or DEFAULT_TARGET_PAYMENT
    coverage_30 = float(payload.get("coverage_ratio_30d", 0))
    coverage_14 = float(payload.get("coverage_ratio_14d", coverage_30))
    downtime = float(payload.get("downtime_hours_30d", 0))
    activity = float(payload.get("activity_drop_pct", 0))
    arrears = float(payload.get("arrears_amount", 0))
    exposure = float(payload.get("exposure_after_transfer", arrears))
    seatbelt = float(payload.get("seatbelt_off_rate_30d", 0))
    high_speed = float(payload.get("high_speed_ratio_30d", 0))
    idle_ratio = float(payload.get("idle_hours_ratio_30d", 0))
    distance = float(payload.get("distance_km_30d", 0))
    after_hours_distance = float(payload.get("after_hours_distance_km_30d", 0))

    coverage_gap = max(0.0, 1 - max(0.0, min(1.0, coverage_30)))
    downtime_norm = max(0.0, min(1.0, downtime / 72))
    activity_norm = max(0.0, min(1.0, activity))
    idle_norm = max(0.0, min(1.0, idle_ratio / max(IDLE_RATIO_THRESHOLD, 1e-6)))
    speed_norm = max(0.0, min(1.0, high_speed / max(SAFETY_SPEED_THRESHOLD, 1e-6)))
    seatbelt_norm = max(0.0, min(1.0, seatbelt / max(SAFETY_SEATBELT_THRESHOLD, 1e-6)))
    after_hours_ratio = 0.0
    if distance > 0:
        after_hours_ratio = max(0.0, min(1.0, after_hours_distance / distance))
    after_hours_norm = max(
        0.0,
        min(1.0, after_hours_ratio / max(AFTER_HOURS_RATIO_THRESHOLD, 1e-6)),
    )
    exposure_ratio = 0.0
    if tp > 0:
        exposure_ratio = max(0.0, min(1.0, arrears / tp))

    risk = round(
        0.30 * coverage_gap
        + 0.20 * downtime_norm
        + 0.12 * activity_norm
        + 0.10 * idle_norm
        + 0.08 * speed_norm
        + 0.07 * seatbelt_norm
        + 0.05 * after_hours_norm
        + 0.08 * exposure_ratio,
        3,
    )

    safety_flag = (seatbelt_norm >= 1) or (speed_norm >= 1)
    needs_protection = int(
        (risk > 0.45)
        or (coverage_14 < 0.65)
        or (exposure > 2500)
        or safety_flag
        or (idle_norm >= 1)
    )

    if safety_flag:
        scenario = "safety-coaching"
    elif downtime > 72 and coverage_30 < 0.5:
        scenario = "restructure-full"
    elif coverage_30 < 0.6:
        scenario = "restructure-light"
    elif idle_norm >= 1:
        scenario = "idle-rebalance"
    elif needs_protection == 0:
        scenario = "monitor"
    else:
        scenario = "advisor-review"

    whatsapp_segment = np.select(
        [safety_flag, needs_protection == 1],
        ["PIA_SEGUIMIENTO", "PIA_OPCIONES"],
        default="FOLLOW_UP",
    )

    return SimulationResult(
        placa=str(payload.get("placa", "UNKNOWN")),
        risk_score=risk,
        needs_protection=needs_protection,
        suggested_scenario=scenario,
        whatsapp_segment=whatsapp_segment,
    )
