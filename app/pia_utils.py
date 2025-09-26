"""Utilities for PIA synthetic dataset generation and serving."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_TARGET_PAYMENT = 11_000.0


def _candidate_snapshot_paths() -> list[Path]:
    cwd = Path.cwd()
    candidates = [
        Path("data/hase/consumos_snapshot_latest.csv.gz"),
        cwd / "data" / "hase" / "consumos_snapshot_latest.csv.gz",
        Path("conductores/data/hase/consumos_snapshot_latest.csv.gz"),
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
        Path("data/pia/pia_features.csv"),
        cwd / "data" / "pia" / "pia_features.csv",
        Path("conductores/data/pia/pia_features.csv"),
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
    raise FileNotFoundError("No se encontró el snapshot de consumos para PIA")


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

    snapshot_df["risk_score"] = (
        0.5 * coverage_gap + 0.3 * downtime_norm + 0.2 * activity_norm
    ).round(3)

    snapshot_df["needs_protection"] = (
        (snapshot_df["risk_score"] > 0.45)
        | (snapshot_df["coverage_ratio_14d"] < 0.65)
        | (snapshot_df["exposure_after_transfer"] > 2500)
    ).astype(int)

    conditions = [
        (snapshot_df["needs_protection"] == 0),
        (snapshot_df["downtime_hours_30d"] > 72)
        & (snapshot_df["coverage_ratio_30d"] < 0.5),
        (snapshot_df["coverage_ratio_30d"] < 0.6),
    ]
    scenarios = [
        "monitor",
        "restructure-full",
        "restructure-light",
    ]
    snapshot_df["suggested_scenario"] = np.select(
        conditions, scenarios, default="advisor-review"
    )

    snapshot_df["whatsapp_segment"] = np.where(
        snapshot_df["needs_protection"] == 1,
        "PIA_OPCIONES",
        "FOLLOW_UP",
    )

    useful_cols = [
        "plaza_limpia",
        "placa",
        "fecha_dia",
        "coverage_ratio_14d",
        "coverage_ratio_30d",
        "downtime_hours_30d",
        "activity_drop_pct",
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
    raise FileNotFoundError("No se encontró el dataset PIA (data/pia/pia_features.csv)")


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

    coverage_gap = max(0.0, 1 - max(0.0, min(1.0, coverage_30)))
    downtime_norm = max(0.0, min(1.0, downtime / 72))
    activity_norm = max(0.0, min(1.0, activity))

    risk = round(0.5 * coverage_gap + 0.3 * downtime_norm + 0.2 * activity_norm, 3)
    needs_protection = int(
        (risk > 0.45)
        or (coverage_14 < 0.65)
        or (exposure > 2500)
    )
    if needs_protection == 0:
        scenario = "monitor"
    elif downtime > 72 and coverage_30 < 0.5:
        scenario = "restructure-full"
    elif coverage_30 < 0.6:
        scenario = "restructure-light"
    else:
        scenario = "advisor-review"

    whatsapp_segment = "PIA_OPCIONES" if needs_protection else "FOLLOW_UP"

    return SimulationResult(
        placa=str(payload.get("placa", "UNKNOWN")),
        risk_score=risk,
        needs_protection=needs_protection,
        suggested_scenario=scenario,
        whatsapp_segment=whatsapp_segment,
    )

