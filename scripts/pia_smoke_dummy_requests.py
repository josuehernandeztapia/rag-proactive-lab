#!/usr/bin/env python3
"""Smoke-test `/pia/protection/evaluate` using the dummy contracts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main import app  # noqa: E402
from scripts.pia_generate_dummy_outcomes import (  # noqa: E402
    DEFAULT_CONTRACTS_PATH,
    FinancialProfile,
    load_contracts,
    select_financials,
)

client = TestClient(app)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Invoke /pia/protection/evaluate for each dummy contract")
    parser.add_argument("--contracts", type=Path, default=DEFAULT_CONTRACTS_PATH, help="CSV with dummy protection contracts")
    parser.add_argument("--market", default="edomex", help="Market value sent in each request")
    parser.add_argument("--risk-band", default="medio", help="Risk band attached when log_outcome is enabled")
    parser.add_argument("--log-outcome", action="store_true", help="Request the API to log outcomes using decision_* fields")
    parser.add_argument("--fail-on-error", action="store_true", help="Exit with code 1 if any request fails")
    return parser.parse_args()


def build_payload(contract, profile: FinancialProfile, market: str, log_outcome: bool, risk_band: str) -> Dict[str, object]:
    payload: Dict[str, object] = {
        "market": market,
        "balance": profile.balance,
        "payment": profile.payment,
        "term_months": profile.term_months,
        "plan_type": contract.plan_type,
        "protections_used": contract.protections_used,
        "protections_allowed": contract.protections_allowed,
        "metadata": {"placa": contract.placa, "source": "pia_smoke_dummy_requests"},
        "requires_manual_review": contract.requires_manual_review,
        "has_delinquency_flag": contract.status.lower() == "expired",
    }
    if log_outcome:
        payload.update(
            {
                "log_outcome": True,
                "decision_placa": contract.placa,
                "decision_risk_band": risk_band,
                "decision_action": "evaluate_protection",
                "decision_reason": "Smoke synthetic request",
            }
        )
    return payload


def main() -> int:
    args = parse_args()
    contracts = load_contracts(args.contracts)
    if not contracts:
        print(f"No contracts found in {args.contracts}", file=sys.stderr)
        return 1

    total = len(contracts)
    successes = 0
    failure = False
    viable_counts: List[int] = []

    for idx, contract in enumerate(contracts):
        profile = select_financials(contract.plan_type, idx)
        payload = build_payload(contract, profile, args.market, args.log_outcome, args.risk_band)
        response = client.post("/pia/protection/evaluate", json=payload)
        if response.status_code != 200:
            failure = True
            print(f"[{contract.placa}] Error {response.status_code}: {response.text}")
            continue
        data = response.json()
        viable = data.get("viable", [])
        viable_counts.append(len(viable))
        successes += 1
        manual_count = sum(1 for item in viable if item.get("requires_manual_review"))
        top_type = viable[0]["type"] if viable else "N/A"
        print(
            f"[{contract.placa}] status={contract.status} plan={contract.plan_type} → "
            f"viable={len(viable)} (manual={manual_count}) top={top_type}"
        )

    print(f"Contratos evaluados: {successes}/{total}")
    if viable_counts:
        print(f"Promedio escenarios viables: {sum(viable_counts) / len(viable_counts):.2f}")
        zero_count = sum(1 for value in viable_counts if value == 0)
        print(f"Contratos sin escenarios viables: {zero_count}")

    if failure and args.fail_on_error:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
