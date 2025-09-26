#!/usr/bin/env python3
"""CLI helper to evaluate protection scenarios using the equilibrium engine.

Usage:
  python3 agents/pia/scripts/evaluate_protection_scenarios.py \
      --market edomex --balance 520000 --payment 18900 --term 48 \
      --restructures-used 0 --plan-type proteccion_total --protections-used 1
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import sys
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.pia.src.contracts import get_contract_for_placa
from agents.pia.src.tir_equilibrium_engine import (
    ProtectionContext,
    evaluate_protection_scenarios,
    get_default_policy,
    select_viable_scenarios,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate protection scenarios keeping IRR above target")
    parser.add_argument("--market", default="aguascalientes")
    parser.add_argument("--balance", type=float, required=True)
    parser.add_argument("--payment", type=float, required=True)
    parser.add_argument("--term", type=int, required=True, help="Remaining term in months")
    parser.add_argument("--restructures-used", type=int, default=0)
    parser.add_argument("--restructures-allowed", type=int, default=None)
    parser.add_argument("--plan-type", default=None)
    parser.add_argument("--protections-used", type=int, default=None)
    parser.add_argument("--protections-allowed", type=int, default=None)
    parser.add_argument("--placa", default=None, help="Placa para buscar contratos dummy si faltan datos")
    parser.add_argument("--manual-review", dest="manual_review", action="store_true")
    parser.add_argument("--no-manual-review", dest="manual_review", action="store_false")
    parser.set_defaults(manual_review=None)
    parser.add_argument("--consumption-gap", action="store_true")
    parser.add_argument("--fault-alert", action="store_true")
    parser.add_argument("--delinquency", action="store_true")
    parser.add_argument("--promise-break", action="store_true")
    parser.add_argument("--no-telematics", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    contract = None
    needs_contract = (
        args.plan_type is None
        or args.protections_allowed is None
        or args.protections_used is None
        or args.manual_review is None
    )
    if needs_contract and args.placa:
        contract = get_contract_for_placa(args.placa)

    plan_type = args.plan_type or (contract.plan_type if contract else None)
    protections_allowed = (
        args.protections_allowed
        if args.protections_allowed is not None
        else (contract.protections_allowed if contract else None)
    )
    protections_used = (
        args.protections_used
        if args.protections_used is not None
        else (contract.protections_used if contract else 0)
    )
    manual_review = args.manual_review
    if manual_review is None:
        manual_review = contract.requires_manual_review if contract else False

    context = ProtectionContext(
        market=args.market,
        balance=args.balance,
        payment=args.payment,
        term_months=args.term,
        restructures_used=args.restructures_used,
        restructures_allowed=args.restructures_allowed,
        plan_type=plan_type,
        protections_used=protections_used,
        protections_allowed=protections_allowed,
        requires_manual_review=manual_review,
        has_consumption_gap=args.consumption_gap,
        has_fault_alert=args.fault_alert,
        has_delinquency_flag=args.delinquency,
        has_recent_promise_break=args.promise_break,
        telematics_ok=not args.no_telematics,
    )

    policy = get_default_policy()
    scenarios = evaluate_protection_scenarios(context, policy)
    viable = select_viable_scenarios(scenarios)

    payload: Dict[str, Any] = {
        "context": context.__dict__,
        "policy": asdict(policy),
        "contract_lookup": {
            "placa": args.placa,
            "found": bool(contract),
            "plan_type": getattr(contract, "plan_type", None) if contract else None,
            "status": getattr(contract, "status", None) if contract else None,
            "valid_until": getattr(contract, "valid_until", None) if contract else None,
            "reset_cycle_days": getattr(contract, "reset_cycle_days", None) if contract else None,
            "requires_manual_review": getattr(contract, "requires_manual_review", None) if contract else None,
        },
        "results": [scenario.to_dict() for scenario in scenarios],
        "viable": [scenario.to_dict() for scenario in viable],
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
