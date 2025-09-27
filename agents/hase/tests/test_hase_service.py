from __future__ import annotations

import csv
from pathlib import Path

from agents.hase.src.service import score_payload

ROOT = Path(__file__).resolve().parents[3]
STATES_PATH = ROOT / "data" / "pia" / "synthetic_driver_states.csv"


def _load_sample() -> dict[str, str]:
    with STATES_PATH.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            return row
    raise RuntimeError("synthetic_driver_states.csv está vacío; ejecuta pia_seed_synthetic_portfolio.py primero")


def test_score_payload_uses_portfolio_snapshot() -> None:
    state = _load_sample()
    payload = {
        "placa": state["placa"],
        "expected_payment": float(state["expected_payment"]),
        "coverage_ratio_30d": float(state["coverage_ratio_30d"]),
        "coverage_ratio_14d": float(state["coverage_ratio_14d"]),
        "downtime_hours_30d": float(state["downtime_hours_30d"]),
        "arrears_amount": float(state["arrears_amount"]),
        "bank_transfer": float(state["bank_transfer"]),
        "gnv_credit_30d": float(state["gnv_credit_30d"]),
    }
    score = score_payload(payload)

    assert score.placa == state["placa"].upper()
    assert "protections_remaining" in score.features
    assert score.metadata.get("used_snapshot") is True
