"""High-level service helper around the TIR equilibrium engine."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .outcomes import record_outcome
from .rules import PIADecision
from .tir_equilibrium_engine import (
    ProtectionContext,
    ProtectionPolicy,
    evaluate_protection_scenarios,
    get_default_policy,
    select_viable_scenarios,
)


def evaluate_scenarios(
    context: ProtectionContext,
    policy: Optional[ProtectionPolicy] = None,
    *,
    decision: Optional[PIADecision] = None,
    log_outcome: bool = False,
    outcome_label: str = "proposed_protection",
    notes: str = "",
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return all scenarios (and viable subset). Optionally log first viable."""

    active_policy = policy or get_default_policy()
    scenarios = evaluate_protection_scenarios(context, active_policy)
    viable = select_viable_scenarios(scenarios)

    if log_outcome and decision and viable:
        top = viable[0]
        meta = metadata.copy() if metadata else {}
        meta.update(
            {
                "protection_scenario": top.to_dict(),
                "equilibrium_policy": {
                    "tir_tolerance_bps": active_policy.tir_tolerance_bps,
                    "max_restructures": active_policy.max_restructures,
                },
                "protection_plan": {
                    "plan_type": context.plan_type,
                    "protections_used": context.protections_used,
                    "protections_allowed": context.protections_allowed,
                    "status": context.contract_status,
                    "valid_until": context.contract_valid_until,
                    "reset_cycle_days": context.contract_reset_cycle_days,
                    "requires_manual_review": context.requires_manual_review,
                },
            }
        )
        record_outcome(
            decision,
            outcome_label,
            plaza=context.market,
            notes=notes,
            metadata=meta,
        )

    return {
        "context": context.__dict__,
        "policy": active_policy.__dict__,
        "scenarios": [scenario.to_dict() for scenario in scenarios],
        "viable": [scenario.to_dict() for scenario in viable],
    }


__all__ = ["evaluate_scenarios"]
