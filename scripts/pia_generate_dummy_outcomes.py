#!/usr/bin/env python3
"""Generate synthetic protection outcomes from dummy contracts.

The script loads `data/pia/protection_contracts_dummy.csv`, evaluates the
protection equilibrium for each contract, logs the top viable scenario and then
runs the same aggregation pipeline exposed via `make pia-aggregate`.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.pia.src.contracts import ProtectionContract  # noqa: E402
from agents.pia.src.outcomes import (  # noqa: E402
    DEFAULT_LOG_PATH,
    aggregate_outcomes,
    load_outcomes,
)
from agents.pia.src.rules import PIADecision, decide_action  # noqa: E402
from agents.pia.src.tir_equilibrium_engine import (  # noqa: E402
    ProtectionContext,
    get_default_policy,
)
from agents.pia.src.tir_equilibrium_service import evaluate_scenarios  # noqa: E402
from agents.hase.src.service import score_payload  # noqa: E402

DEFAULT_SUMMARY_PATH = ROOT / "reports" / "pia_plan_summary.csv"
DEFAULT_FEATURES_PATH = ROOT / "data" / "hase" / "pia_outcomes_features.csv"
DEFAULT_CONTRACTS_PATH = ROOT / "data" / "pia" / "synthetic_contracts.csv"
DEFAULT_STATES_PATH = ROOT / "data" / "pia" / "synthetic_driver_states.csv"


@dataclass(frozen=True)
class FinancialProfile:
    balance: float
    payment: float
    term_months: int


FINANCIAL_PROFILES = {
    "proteccion_total": FinancialProfile(balance=520_000, payment=19_000, term_months=48),
    "proteccion_basica": FinancialProfile(balance=470_000, payment=16_800, term_months=44),
    "proteccion_light": FinancialProfile(balance=360_000, payment=13_500, term_months=36),
    "proteccion_caduca": FinancialProfile(balance=455_000, payment=17_200, term_months=40),
    "proteccion_ilimitada": FinancialProfile(balance=610_000, payment=20_500, term_months=60),
}
DEFAULT_PROFILE = FinancialProfile(balance=500_000, payment=18_000, term_months=48)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic protection outcomes and aggregate them")
    parser.add_argument("--contracts", type=Path, default=DEFAULT_CONTRACTS_PATH, help="CSV with dummy protection contracts")
    parser.add_argument("--states", type=Path, default=DEFAULT_STATES_PATH, help="CSV con estados y telemetría sintética")
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG_PATH, help="Where to store outcome log entries")
    parser.add_argument("--market", default="edomex", help="Market assigned to generated contexts")
    parser.add_argument("--notes", default="Synthetic lab outcome", help="Notes to attach to recorded outcomes")
    parser.add_argument("--windows", type=int, nargs="*", default=[30, 90, 180], help="Windows (days) for aggregation pivots")
    parser.add_argument("--features-out", type=Path, default=DEFAULT_FEATURES_PATH, help="Destination for aggregated features CSV")
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY_PATH, help="Destination for plan summary CSV")
    parser.add_argument("--reset-log", action="store_true", help="Delete the target log before generating outcomes")
    parser.add_argument("--skip-aggregate", action="store_true", help="Skip aggregation step after generating outcomes")
    return parser.parse_args()


def load_contracts(path: Path) -> List[ProtectionContract]:
    contracts: List[ProtectionContract] = []
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el CSV de contratos dummy: {path}")
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            placa = (row.get("placa") or "").strip()
            if not placa:
                continue
            try:
                protections_allowed = int(row.get("protections_allowed", 0) or 0)
            except (TypeError, ValueError):
                protections_allowed = 0
            try:
                protections_used = int(row.get("protections_used", 0) or 0)
            except (TypeError, ValueError):
                protections_used = 0
            status = (row.get("status") or "active").strip()
            valid_until = (row.get("valid_until") or "").strip() or None
            try:
                reset_cycle_days = int(row.get("reset_cycle_days", 0) or 0)
            except (TypeError, ValueError):
                reset_cycle_days = None
            requires_manual_review = str(row.get("requires_manual_review") or "").strip().lower() in {"1", "true", "yes", "y"}
            contracts.append(
                ProtectionContract(
                    placa=placa,
                    plan_type=(row.get("plan_type") or "unknown").strip(),
                    protections_allowed=protections_allowed,
                    protections_used=protections_used,
                    status=status or "active",
                    valid_until=valid_until,
                    reset_cycle_days=reset_cycle_days,
                    requires_manual_review=requires_manual_review,
                )
            )
    return contracts


def load_states(path: Path) -> Dict[str, Dict[str, str]]:
    states: Dict[str, Dict[str, str]] = {}
    if not path.exists():
        return states
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            placa = (row.get("placa") or "").strip().upper()
            if not placa:
                continue
            states[placa] = row
    return states


def _to_float(value: Optional[str], default: float = 0.0) -> float:
    if value is None or value == "":
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _to_int(value: Optional[str], default: int = 0) -> int:
    if value is None or value == "":
        return int(default)
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _to_bool(value: Optional[str | int | float | bool], default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def select_financials(plan_type: str, index: int) -> FinancialProfile:
    key = plan_type.strip().lower()
    profile = FINANCIAL_PROFILES.get(key)
    if profile:
        return profile
    # Slightly perturb the fallback profile so each record differs.
    balance = DEFAULT_PROFILE.balance * (1 + 0.02 * index)
    payment = DEFAULT_PROFILE.payment * (1 + 0.015 * index)
    term = DEFAULT_PROFILE.term_months
    return FinancialProfile(balance=balance, payment=payment, term_months=term)


def iterate_contracts(path: Path) -> Iterator[Tuple[int, ProtectionContract]]:
    for idx, contract in enumerate(load_contracts(path)):
        yield idx, contract


def build_context(
    contract: ProtectionContract,
    profile: FinancialProfile,
    market: str,
    state: Optional[Dict[str, str]],
) -> ProtectionContext:
    if state:
        market_value = (state.get("market") or market).strip() or market
        balance = _to_float(state.get("balance"), profile.balance)
        payment = _to_float(state.get("payment"), profile.payment)
        term_months = _to_int(state.get("term_months"), profile.term_months)
        protections_allowed = _to_int(state.get("protections_allowed"), contract.protections_allowed)
        protections_used = _to_int(state.get("protections_used"), contract.protections_used)
        status = (state.get("contract_status") or contract.status or "active").strip()
        valid_until = state.get("contract_valid_until") or contract.valid_until
        reset_cycle_days_raw = _to_int(state.get("contract_reset_cycle_days"), contract.reset_cycle_days or 0)
        reset_cycle_days = reset_cycle_days_raw or None
        requires_manual = _to_bool(state.get("requires_manual_review"), contract.requires_manual_review)
        has_consumption_gap = bool(_to_int(state.get("hase_consumption_gap_flag"), 0))
        has_fault_alert = bool(_to_int(state.get("hase_fault_alert_flag"), 0))
        telemetry_ok = bool(_to_int(state.get("hase_telemetry_ok_flag"), 1))
        scenario_label = (state.get("scenario") or "").strip().lower()
        has_recent_promise = bool(_to_int(state.get("has_recent_promise_break"), 0))
        has_delinquency = status.lower() == "expired" or scenario_label == "delinquency"
    else:
        market_value = market
        balance = profile.balance
        payment = profile.payment
        term_months = profile.term_months
        protections_allowed = contract.protections_allowed
        protections_used = contract.protections_used
        status = contract.status
        valid_until = contract.valid_until
        reset_cycle_days = contract.reset_cycle_days
        requires_manual = contract.requires_manual_review
        has_consumption_gap = False
        has_fault_alert = False
        telemetry_ok = True
        has_recent_promise = False
        has_delinquency = contract.status.lower() == "expired"

    return ProtectionContext(
        market=market_value,
        balance=balance,
        payment=payment,
        term_months=term_months,
        restructures_used=0,
        restructures_allowed=None,
        plan_type=contract.plan_type,
        protections_used=protections_used,
        protections_allowed=protections_allowed,
        contract_status=status,
        contract_valid_until=valid_until,
        contract_reset_cycle_days=reset_cycle_days,
        requires_manual_review=requires_manual,
        has_consumption_gap=has_consumption_gap,
        has_fault_alert=has_fault_alert,
        has_delinquency_flag=has_delinquency,
        has_recent_promise_break=has_recent_promise,
        telematics_ok=telemetry_ok,
    )


def prepare_decision(
    contract: ProtectionContract,
    profile: FinancialProfile,
    state: Optional[Dict[str, str]],
) -> Tuple[PIADecision, Dict[str, Any]]:
    expected_payment = profile.payment
    payload: Dict[str, Any] = {
        "placa": contract.placa,
        "expected_payment": expected_payment,
        "coverage_ratio_30d": 0.0,
        "coverage_ratio_14d": 0.0,
        "downtime_hours_30d": 0.0,
        "bank_transfer": 0.0,
        "gnv_credit_30d": 0.0,
        "arrears_amount": 0.0,
        "last_protection_at": None,
        "protections_allowed": contract.protections_allowed,
        "protections_used": contract.protections_used,
    }

    if state:
        payload.update(
            {
                "expected_payment": _to_float(state.get("expected_payment"), expected_payment),
                "coverage_ratio_30d": _to_float(state.get("coverage_ratio_30d"), 0.0),
                "coverage_ratio_14d": _to_float(state.get("coverage_ratio_14d"), 0.0),
                "downtime_hours_30d": _to_float(state.get("downtime_hours_30d"), 0.0),
                "bank_transfer": _to_float(state.get("bank_transfer"), 0.0),
                "gnv_credit_30d": _to_float(state.get("gnv_credit_30d"), 0.0),
                "gnv_credit_14d": _to_float(state.get("gnv_credit_14d"), 0.0),
                "gnv_credit_7d": _to_float(state.get("gnv_credit_7d"), 0.0),
                "arrears_amount": _to_float(state.get("arrears_amount"), 0.0),
                "distance_km_30d": _to_float(state.get("distance_km_30d"), 0.0),
                "engine_hours_30d": _to_float(state.get("engine_hours_30d"), 0.0),
                "fault_events_30d": _to_float(state.get("fault_events_30d"), 0.0),
                "cash_collection": _to_float(state.get("cash_collection"), 0.0),
                "observed_payment": _to_float(state.get("observed_payment"), 0.0),
                "avg_30d_litros": _to_float(state.get("avg_30d_litros"), 0.0),
                "last_protection_at": state.get("last_protection_at"),
                "suggested_scenario": state.get("scenario"),
                "hase_consumption_gap_flag": _to_int(state.get("hase_consumption_gap_flag"), 0),
                "hase_fault_alert_flag": _to_int(state.get("hase_fault_alert_flag"), 0),
                "hase_telemetry_ok_flag": _to_int(state.get("hase_telemetry_ok_flag"), 1),
                "protections_allowed": _to_int(state.get("protections_allowed"), contract.protections_allowed),
                "protections_used": _to_int(state.get("protections_used"), contract.protections_used),
            }
        )
        scenario_label = (state.get("scenario") or "").strip().lower()
        if scenario_label == "delinquency" and _to_int(state.get("has_recent_promise_break"), 0):
            payload["intent"] = "payment_promise"
        elif scenario_label == "baseline" and _to_float(state.get("arrears_amount"), 0.0) < 0:
            payload["intent"] = "advance_payment"

    hase_score = score_payload(payload)
    payload["risk_score"] = hase_score.risk_score

    decision = decide_action(payload)
    decision.details.update(
        {
            "bank_transfer": payload.get("bank_transfer", 0.0),
            "cash_collection": payload.get("cash_collection", 0.0),
            "gnv_credit_30d": payload.get("gnv_credit_30d", 0.0),
            "gnv_credit_14d": payload.get("gnv_credit_14d", 0.0),
            "gnv_credit_7d": payload.get("gnv_credit_7d", 0.0),
            "downtime_hours_30d": payload.get("downtime_hours_30d", 0.0),
            "scenario_hint": payload.get("suggested_scenario"),
        }
    )
    if state:
        decision.details.setdefault("market", state.get("market"))
    else:
        decision.details.setdefault("market", "synthetic_lab")

    metadata: Dict[str, Any] = {
        "hase_score": hase_score.to_dict(),
    }
    if state:
        metadata["synthetic_state"] = state
    return decision, metadata


def log_outcome(
    context: ProtectionContext,
    decision: PIADecision,
    *,
    plan_label: str,
    notes: str,
    log_path: Path,
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[dict]:
    policy = get_default_policy()
    metadata_payload: Dict[str, Any] = {
        "channel": "synthetic_lab",
        "source": "pia_generate_dummy_outcomes",
    }
    if metadata:
        metadata_payload.update(metadata)
    result = evaluate_scenarios(
        context,
        policy,
        decision=decision,
        log_outcome=True,
        outcome_label="synthetic_protection",
        notes=notes,
        metadata=metadata_payload,
    )
    viable = result.get("viable", [])
    if viable:
        top = viable[0]
        print(
            f"[{decision.placa}] {plan_label} | {decision.action} → {top['type']} (manual={top.get('requires_manual_review', False)})",
            flush=True,
        )
        return top
    print(f"[{decision.placa}] {plan_label} → sin escenarios viables", flush=True)
    return None


def run_aggregation(log_path: Path, features_out: Path, summary_out: Path, windows: List[int]) -> None:
    try:
        df = load_outcomes(log_path)
    except FileNotFoundError:
        print(f"No outcome log found at {log_path}, skipping aggregation", file=sys.stderr)
        return
    if df.empty:
        print(f"Outcome log {log_path} está vacío; sin agregados", file=sys.stderr)
        return
    aggregated = aggregate_outcomes(df, windows=tuple(windows))
    if aggregated.empty:
        print("aggregate_outcomes regresó vacío", file=sys.stderr)
        return
    if "protections_remaining" not in aggregated.columns:
        aggregated["protections_remaining"] = 0
    if "last_plan_status" not in aggregated.columns:
        aggregated["last_plan_status"] = ""
    if "last_plan_requires_manual_review" not in aggregated.columns:
        aggregated["last_plan_requires_manual_review"] = False
    aggregated["protections_flag_negative"] = aggregated["protections_remaining"].fillna(0) < 0
    aggregated["protections_flag_expired"] = aggregated["last_plan_status"].fillna("").str.lower().eq("expired")
    aggregated["protections_flag_manual"] = aggregated["last_plan_requires_manual_review"].fillna(False).astype(bool)

    features_out.parent.mkdir(parents=True, exist_ok=True)
    aggregated.to_csv(features_out, index=False)
    print(f"Características agregadas guardadas en {features_out} ({len(aggregated)} placas)")

    if summary_out:
        summary_out.parent.mkdir(parents=True, exist_ok=True)
        if "last_plan_type" not in aggregated.columns:
            aggregated["last_plan_type"] = "unknown"
        plan_summary = (
            aggregated.groupby("last_plan_type")
            .agg(
                contratos=("placa", "nunique"),
                protecciones_restantes_promedio=("protections_remaining", "mean"),
                protecciones_restantes_mediana=("protections_remaining", "median"),
                contratos_manual=("protections_flag_manual", "sum"),
                contratos_expirados=("protections_flag_expired", "sum"),
                contratos_negative=("protections_flag_negative", "sum"),
            )
            .reset_index()
            .rename(columns={"last_plan_type": "plan_type"})
        )
        plan_summary.to_csv(summary_out, index=False)
        print(f"Resumen por plan guardado en {summary_out}")


def maybe_reset_log(log_path: Path) -> None:
    if log_path.exists():
        log_path.unlink()
        print(f"Log reiniciado: {log_path}")


def main() -> int:
    args = parse_args()
    if args.reset_log:
        maybe_reset_log(args.log)

    states = load_states(args.states)

    total_viable = 0
    for index, contract in iterate_contracts(args.contracts):
        profile = select_financials(contract.plan_type, index)
        state = states.get(contract.placa.upper())
        context = build_context(contract, profile, args.market, state)
        decision, metadata = prepare_decision(contract, profile, state)
        plan_label = context.plan_type or contract.plan_type or 'unknown'
        result = log_outcome(
            context,
            decision,
            plan_label=plan_label,
            notes=args.notes,
            log_path=args.log,
            metadata=metadata,
        )
        if result is not None:
            total_viable += 1

    print(f"Escenarios viables registrados: {total_viable}")

    if not args.skip_aggregate:
        run_aggregation(args.log, args.features_out, args.summary_out, args.windows)
    else:
        print("Omitiendo agregación por bandera --skip-aggregate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
