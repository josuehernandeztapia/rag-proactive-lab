"""Laboratory scoring stub for the HASE engine.

This module provides a deterministic-but-transparent scorer that the PIA agent can
use while the real-time HASE service is under construction. It combines whatever
metrics are present in the payload with optional aggregates exported by
``agents/hase/scripts/aggregate_pia_outcomes.py``. The goal is to remove mock
risk scores from the agent chain while keeping the logic simple enough for lab
validation.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

ROOT_DIR = Path(__file__).resolve().parents[3]

_AGGREGATE_CANDIDATES = (
    ROOT_DIR / "data" / "hase" / "pia_outcomes_features.csv",
    Path("data/hase/pia_outcomes_features.csv"),
)
_PORTFOLIO_CANDIDATES = (
    ROOT_DIR / "data" / "pia" / "synthetic_driver_states.csv",
    Path("data/pia/synthetic_driver_states.csv"),
)


def _coerce_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    if isinstance(value, (int, float)):
        return float(value)
    try:
        cleaned = str(value).strip()
        if not cleaned:
            return float(default)
        return float(cleaned)
    except (TypeError, ValueError):
        return float(default)


def _coerce_int(value: Any, default: int = 0) -> int:
    if value is None:
        return int(default)
    if isinstance(value, int):
        return value
    try:
        cleaned = str(value).strip()
        if not cleaned:
            return int(default)
        return int(float(cleaned))
    except (TypeError, ValueError):
        return int(default)


@dataclass(frozen=True)
class HaseScore:
    placa: str
    risk_score: float
    probability_default: float
    features: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "placa": self.placa,
            "risk_score": round(self.risk_score, 3),
            "probability_default": round(self.probability_default, 3),
            "features": self.features,
            "metadata": self.metadata,
        }


@lru_cache(maxsize=1)
def _load_outcomes_snapshot() -> Dict[str, Dict[str, Any]]:
    for path in _AGGREGATE_CANDIDATES:
        try:
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                rows = {row.get("placa", "").upper(): row for row in reader if row.get("placa")}
            if rows:
                return rows
        except Exception:
            continue
    return {}


@lru_cache(maxsize=1)
def _load_portfolio_snapshot() -> Dict[str, Dict[str, Any]]:
    for path in _PORTFOLIO_CANDIDATES:
        try:
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                rows = {row.get("placa", "").upper(): row for row in reader if row.get("placa")}
            if rows:
                return rows
        except Exception:
            continue
    return {}


def _combine_snapshots(placa: str) -> Dict[str, Any]:
    combined: Dict[str, Any] = {}
    portfolio = _load_portfolio_snapshot().get(placa)
    outcomes = _load_outcomes_snapshot().get(placa)
    if portfolio:
        combined.update(portfolio)
    if outcomes:
        combined.update({key: value for key, value in outcomes.items() if key not in combined})
    return combined


def _collect_features(payload: Dict[str, Any], snapshot: Optional[Dict[str, Any]]) -> Dict[str, float]:
    coverage_30 = _coerce_float(payload.get("coverage_ratio_30d"), 0.0)
    coverage_14 = _coerce_float(payload.get("coverage_ratio_14d"), coverage_30)
    downtime_hours = _coerce_float(payload.get("downtime_hours_30d"), 0.0)
    arrears_amount = _coerce_float(payload.get("arrears_amount"), 0.0)
    expected_payment = _coerce_float(payload.get("expected_payment"), 11_000.0)
    bank_transfer = _coerce_float(payload.get("bank_transfer"), 0.0)
    gnv_credit = _coerce_float(payload.get("gnv_credit_30d"), 0.0)

    features: Dict[str, float] = {
        "coverage_ratio_30d": coverage_30,
        "coverage_ratio_14d": coverage_14,
        "downtime_hours_30d": downtime_hours,
        "arrears_amount": arrears_amount,
        "expected_payment": expected_payment,
        "bank_transfer": bank_transfer,
        "gnv_credit_30d": gnv_credit,
    }

    if snapshot:
        numeric_keys = {
            "protections_remaining",
            "behaviour_tag_payment_refusal_count",
            "downtime_hours_14d",
            "downtime_hours_7d",
            "coverage_ratio_7d",
            "cash_collection",
            "observed_payment",
            "gnv_credit_14d",
            "gnv_credit_7d",
        }
        for key in numeric_keys:
            value = snapshot.get(key)
            if value is None:
                continue
            try:
                features[key] = float(value)
            except (TypeError, ValueError):
                continue
    return features


def _score_from_features(features: Dict[str, float]) -> float:
    coverage_gap = max(0.0, 1.0 - min(features.get("coverage_ratio_14d", 0.0), 1.0))
    payment_gap = max(
        0.0,
        features.get("expected_payment", 0.0)
        - (features.get("bank_transfer", 0.0) + features.get("gnv_credit_30d", 0.0)),
    )
    arrears = max(0.0, features.get("arrears_amount", 0.0))
    expected_payment = max(features.get("expected_payment", 1.0), 1.0)

    arrears_ratio = arrears / expected_payment
    payment_gap_ratio = payment_gap / expected_payment
    downtime_norm = min(features.get("downtime_hours_30d", 0.0) / 120.0, 1.0)

    score = (
        0.45 * coverage_gap
        + 0.25 * payment_gap_ratio
        + 0.20 * downtime_norm
        + 0.10 * min(arrears_ratio, 1.0)
    )
    return min(max(score, 0.0), 1.0)


def _logistic_probability(score: float) -> float:
    return 1.0 / (1.0 + math.exp(-((score * 6.0) - 3.0)))


def score_payload(payload: Dict[str, Any]) -> HaseScore:
    if not isinstance(payload, dict):
        raise TypeError("payload must be a mapping with vehicle metrics")

    placa_raw = payload.get("placa") or payload.get("plate") or "UNKNOWN"
    placa = str(placa_raw).upper()

    snapshot = _combine_snapshots(placa)
    features = _collect_features(payload, snapshot or None)
    risk_score = _score_from_features(features)
    probability_default = _logistic_probability(risk_score)

    metadata: Dict[str, Any] = {
        "source": "hase_lab_stub",
        "used_snapshot": bool(snapshot),
    }
    if snapshot:
        metadata["snapshot_version"] = snapshot.get("window") or snapshot.get("_window")
        metadata["protections_remaining"] = _coerce_int(snapshot.get("protections_remaining"), 0)

    return HaseScore(
        placa=placa,
        risk_score=risk_score,
        probability_default=probability_default,
        features=features,
        metadata=metadata,
    )


def score_driver(placa: str, *, payload: Optional[Dict[str, Any]] = None) -> HaseScore:
    base = payload.copy() if isinstance(payload, dict) else {}
    base.setdefault("placa", placa)
    return score_payload(base)


__all__ = ["HaseScore", "score_payload", "score_driver"]
