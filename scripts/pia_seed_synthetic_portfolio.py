#!/usr/bin/env python3
"""Seed a synthetic driver portfolio grounded on historical HASE telemetry.

The generator samples the latest records per placa from the HASE training dataset,
perturbs them with realistic noise and assigns each driver a protection plan.

Outputs
-------
- data/pia/synthetic_contracts.csv
- data/pia/synthetic_driver_states.csv

Both files are produced with ~200 financiamientos by default and can be fed into
`pia_generate_dummy_outcomes.py` to simulate the full loop (HASE → PIA → TIR).
"""
from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
HASE_DATA_CANDIDATES: tuple[Path, ...] = (
    DATA_DIR / "hase" / "hase_training_dataset_full.csv.gz",
    DATA_DIR / "hase" / "hase_training_dataset.csv.gz",
)

CONTRACTS_OUT_DEFAULT = DATA_DIR / "pia" / "synthetic_contracts.csv"
STATES_OUT_DEFAULT = DATA_DIR / "pia" / "synthetic_driver_states.csv"


@dataclass(frozen=True)
class PlanConfig:
    name: str
    balance: float
    payment: float
    term_months: int
    protections_allowed: int
    reset_cycle_days: int


PLAN_CATALOG: Dict[str, PlanConfig] = {
    "proteccion_total": PlanConfig("proteccion_total", 520_000, 19_000, 48, 3, 365),
    "proteccion_basica": PlanConfig("proteccion_basica", 470_000, 16_800, 44, 2, 240),
    "proteccion_light": PlanConfig("proteccion_light", 360_000, 13_500, 36, 1, 180),
    "proteccion_caduca": PlanConfig("proteccion_caduca", 455_000, 17_200, 40, 2, 365),
    "proteccion_ilimitada": PlanConfig("proteccion_ilimitada", 610_000, 20_500, 60, 4, 180),
}

PLAN_WEIGHTS: Dict[str, float] = {
    "proteccion_total": 0.32,
    "proteccion_basica": 0.26,
    "proteccion_light": 0.18,
    "proteccion_caduca": 0.14,
    "proteccion_ilimitada": 0.10,
}

SCENARIO_WEIGHTS: Dict[str, float] = {
    "baseline": 0.55,
    "consumption_gap": 0.2,
    "fault_alert": 0.15,
    "delinquency": 0.1,
}

RNG = np.random.default_rng()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic driver metrics grounded on historical telemetry")
    parser.add_argument("--size", type=int, default=200, help="Número de financiamientos a generar")
    parser.add_argument("--seed", type=int, default=42, help="Semilla para reproducibilidad")
    parser.add_argument(
        "--contracts-out",
        type=Path,
        default=CONTRACTS_OUT_DEFAULT,
        help="Ruta de salida para contratos sintéticos",
    )
    parser.add_argument(
        "--states-out",
        type=Path,
        default=STATES_OUT_DEFAULT,
        help="Ruta de salida para estados/telemetría sintética",
    )
    parser.add_argument(
        "--hase-source",
        type=Path,
        default=None,
        help="Ruta explícita al dataset histórico de HASE (CSV/CSV.GZ)",
    )
    return parser.parse_args()


def load_hase_snapshot(path: Path | None) -> pd.DataFrame:
    """Return the latest record per placa from the HASE training dataset."""

    candidate_paths: Iterable[Path]
    if path is not None:
        candidate_paths = (path,)
    else:
        candidate_paths = HASE_DATA_CANDIDATES

    for candidate in candidate_paths:
        if candidate.exists():
            df = pd.read_csv(candidate, parse_dates=["fecha_dia"], low_memory=False)
            break
    else:
        raise FileNotFoundError(
            "No pude encontrar un dataset histórico de HASE. Revisa data/hase/hase_training_dataset_full.csv.gz"
        )

    if "fecha_dia" not in df.columns:
        df["fecha_dia"] = pd.Timestamp.today().normalize()

    df = df.sort_values(["placa", "fecha_dia"]).drop_duplicates("placa", keep="last")
    df = df.reset_index(drop=True)
    return df


def pick_plan(rng: np.random.Generator) -> PlanConfig:
    plans, weights = zip(*PLAN_WEIGHTS.items())
    choice = rng.choice(plans, p=weights)
    return PLAN_CATALOG[str(choice)]


def choose_scenario(rng: np.random.Generator) -> str:
    scenarios, weights = zip(*SCENARIO_WEIGHTS.items())
    return str(rng.choice(scenarios, p=weights))


def _clip(value: float, lo: float, hi: float) -> float:
    return float(min(max(value, lo), hi))


def _clean(value: float | int | str | None, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    try:
        val = float(value)
        if math.isnan(val):
            return float(default)
        return val
    except (TypeError, ValueError):
        return float(default)


def synthesize_records(df: pd.DataFrame, size: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
#if size > len(df): sample with replacement.
    rng = np.random.default_rng(seed)
    sampled = df.sample(n=size, replace=len(df) < size, random_state=seed).reset_index(drop=True)

    contracts: List[Dict[str, object]] = []
    states: List[Dict[str, object]] = []

    today = date.today()
    base_valid_until = today.replace(day=28)

    for idx, row in sampled.iterrows():
        plan = pick_plan(rng)
        scenario = choose_scenario(rng)

        placa = f"LAB-{idx:03d}"
        plaza = str(row.get("plaza_limpia", "EDOMEX")).strip() or "EDOMEX"

        coverage30 = _clip(_clean(row.get("coverage_ratio_30d"), default=0.9), 0.1, 1.6)
        coverage14 = _clip(_clean(row.get("coverage_ratio_14d"), default=coverage30 * 0.85), 0.05, 1.5)
        coverage7 = _clip(_clean(row.get("coverage_ratio_7d"), default=coverage14 * 0.9), 0.02, 1.4)

        downtime30 = _clip(_clean(row.get("downtime_hours_30d"), default=row.get("downtime_hours")), 0.0, 280.0)
        downtime14 = _clip(_clean(row.get("downtime_hours_14d"), default=downtime30 * 0.45), 0.0, 160.0)
        downtime7 = _clip(_clean(row.get("downtime_hours_7d"), default=downtime14 * 0.5), 0.0, 96.0)

        litros30 = _clip(_clean(row.get("litros_30d"), 950.0), 200.0, 3800.0)
        avg_30d_litros = litros30 / 30.0

        expected_payment = plan.payment

        # Payment channel composition.
        if scenario == "delinquency":
            transfer_ratio = rng.uniform(0.2, 0.45)
        elif scenario == "consumption_gap":
            transfer_ratio = rng.uniform(0.35, 0.55)
        elif scenario == "fault_alert":
            transfer_ratio = rng.uniform(0.4, 0.6)
        else:
            transfer_ratio = rng.uniform(0.45, 0.7)

        bank_transfer = round(expected_payment * transfer_ratio, 2)
        # Cash reflects total observed payment shortfall or surplus.
        if scenario == "delinquency":
            effective_ratio = rng.uniform(0.35, 0.85)
        elif scenario == "consumption_gap":
            effective_ratio = rng.uniform(0.75, 0.95)
        elif scenario == "fault_alert":
            effective_ratio = rng.uniform(0.8, 1.0)
        else:
            effective_ratio = rng.uniform(0.95, 1.08)

        observed_payment = _clip(expected_payment * effective_ratio, 5000.0, expected_payment * 1.2)
        cash_collection = max(observed_payment - bank_transfer, 0.0)

        arrears_amount = round(max(0.0, expected_payment - observed_payment), 2)
        if scenario == "baseline" and rng.random() < 0.12:
            arrears_amount = round(-expected_payment * rng.uniform(0.05, 0.12), 2)  # prepago

        gnv_credit_30d = round(expected_payment * coverage30 * rng.uniform(0.85, 1.1), 2)
        gnv_credit_14d = round(expected_payment * coverage14, 2)
        gnv_credit_7d = round(expected_payment * coverage7, 2)

        distance_km_30d = round(litros30 * rng.uniform(2.6, 3.4), 1)
        engine_hours_30d = round(distance_km_30d / rng.uniform(25.0, 32.0), 1)
        fault_events_30d = 0
        if scenario == "fault_alert":
            fault_events_30d = rng.integers(2, 6)
        elif downtime30 > 180:
            fault_events_30d = rng.integers(1, 3)

        hase_consumption_gap_flag = int(scenario == "consumption_gap")
        hase_fault_alert_flag = int(scenario == "fault_alert")
        hase_telemetry_ok_flag = int(downtime14 < 120)

        protections_allowed = plan.protections_allowed
        protections_used = int(rng.integers(0, protections_allowed + 1))
        protections_remaining = protections_allowed - protections_used
        if scenario == "delinquency" and protections_remaining <= 0:
            protections_remaining -= rng.integers(1, 2)
        if scenario == "fault_alert" and protections_remaining == protections_allowed:
            protections_used = max(1, protections_used)
            protections_remaining = protections_allowed - protections_used

        status = "active"
        requires_manual_review = False
        if scenario == "delinquency" and rng.random() < 0.55:
            status = "expired"
            requires_manual_review = True
        elif scenario == "fault_alert" and rng.random() < 0.25:
            requires_manual_review = True
        elif protections_remaining < 0:
            requires_manual_review = True

        has_recent_promise_break = bool(scenario == "delinquency" and rng.random() < 0.6)

        days_offset = int(rng.uniform(18, 120))
        last_protection_at = (today - timedelta(days=days_offset)).isoformat()

        contract = {
            "placa": placa,
            "plan_type": plan.name,
            "protections_allowed": protections_allowed,
            "protections_used": protections_used,
            "status": status,
            "valid_until": (base_valid_until + timedelta(days=int(rng.integers(45, 220)))).isoformat(),
            "reset_cycle_days": plan.reset_cycle_days,
            "requires_manual_review": "yes" if requires_manual_review else "no",
        }
        contracts.append(contract)

        state = {
            "placa": placa,
            "market": plaza,
            "plan_type": plan.name,
            "balance": plan.balance,
            "payment": plan.payment,
            "term_months": plan.term_months,
            "protections_allowed": protections_allowed,
            "protections_used": protections_used,
            "protections_remaining": protections_remaining,
            "contract_status": status,
            "contract_valid_until": contract["valid_until"],
            "contract_reset_cycle_days": plan.reset_cycle_days,
            "requires_manual_review": requires_manual_review,
            "expected_payment": expected_payment,
            "bank_transfer": round(bank_transfer, 2),
            "cash_collection": round(cash_collection, 2),
            "observed_payment": round(observed_payment, 2),
            "arrears_amount": arrears_amount,
            "coverage_ratio_30d": round(coverage30, 3),
            "coverage_ratio_14d": round(coverage14, 3),
            "coverage_ratio_7d": round(coverage7, 3),
            "gnv_credit_30d": gnv_credit_30d,
            "gnv_credit_14d": gnv_credit_14d,
            "gnv_credit_7d": gnv_credit_7d,
            "litros_30d": round(litros30, 2),
            "avg_30d_litros": round(avg_30d_litros, 2),
            "downtime_hours_30d": round(downtime30, 2),
            "downtime_hours_14d": round(downtime14, 2),
            "downtime_hours_7d": round(downtime7, 2),
            "distance_km_30d": round(distance_km_30d, 1),
            "engine_hours_30d": round(engine_hours_30d, 1),
            "fault_events_30d": int(fault_events_30d),
            "hase_consumption_gap_flag": hase_consumption_gap_flag,
            "hase_fault_alert_flag": hase_fault_alert_flag,
            "hase_telemetry_ok_flag": hase_telemetry_ok_flag,
            "has_recent_promise_break": int(has_recent_promise_break),
            "last_protection_at": last_protection_at,
            "scenario": scenario,
            "risk_story": (
                "proteccion_negativa"
                if protections_remaining < 0
                else ("promesa_incumplida" if has_recent_promise_break else scenario)
            ),
        }
        states.append(state)

    return pd.DataFrame(contracts), pd.DataFrame(states)


def main() -> int:
    args = parse_args()
    global RNG
    RNG = np.random.default_rng(args.seed)

    df = load_hase_snapshot(args.hase_source)

    contracts_df, states_df = synthesize_records(df, args.size, args.seed)

    args.contracts_out.parent.mkdir(parents=True, exist_ok=True)
    args.states_out.parent.mkdir(parents=True, exist_ok=True)

    contracts_df.to_csv(args.contracts_out, index=False)
    states_df.to_csv(args.states_out, index=False)

    print(
        f"Generados {len(states_df)} contratos sintéticos → {args.contracts_out.relative_to(ROOT)} / {args.states_out.relative_to(ROOT)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
