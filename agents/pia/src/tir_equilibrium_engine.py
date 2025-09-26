"""Backend engine to evaluate protection scenarios while preserving target IRR.

Ported from the Angular simulators but enriched to operate with real portfolio
signals (consumption gaps, fault alerts, restructure counts).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import yaml


ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = ROOT_DIR / "config" / "financial.yml"

ScenarioType = Literal[
    "DEFER",
    "STEPDOWN",
    "STEPDOWN_DEEP",
    "RECALENDAR",
    "RECALENDAR_PARTIAL",
    "DEFER_STEPDOWN",
    "BALLOON",
    "BALLOON_STAGED",
]


@dataclass(frozen=True)
class ProtectionPolicy:
    """Static policy constraints for protection scenarios."""

    tir_min_by_market: Dict[str, float]
    tir_tolerance_bps: int = 0  # tolerance in basis points
    max_restructures: int = 2
    max_deferral_months: int = 6
    max_stepdown_months: int = 6
    max_stepdown_reduction: float = 0.5  # 50 %
    soft_deferral_cap_with_gap: int = 3
    soft_stepdown_cap_with_gap: float = 0.25


@dataclass(frozen=True)
class ProtectionContext:
    """Current contract state, coverage metadata and behavioural signals."""

    market: str
    balance: float
    payment: float
    term_months: int
    restructures_used: int = 0
    restructures_allowed: Optional[int] = None
    plan_type: Optional[str] = None
    protections_used: int = 0
    protections_allowed: Optional[int] = None
    contract_status: Optional[str] = None
    contract_valid_until: Optional[str] = None
    contract_reset_cycle_days: Optional[int] = None
    requires_manual_review: bool = False
    has_consumption_gap: bool = False
    has_fault_alert: bool = False
    has_delinquency_flag: bool = False
    has_recent_promise_break: bool = False
    telematics_ok: bool = True


@dataclass(frozen=True)
class ScenarioParams:
    type: ScenarioType
    months: int = 0
    reduction_factor: float = 1.0  # only for step-down (0.5 = 50 % payment)
    deferral_months: int = 0
    stepdown_months: int = 0


@dataclass
class ProtectionScenarioResult:
    type: ScenarioType
    params: ScenarioParams
    new_payment: float
    new_term: int
    cash_flows: List[float]
    annual_irr: float
    irr_target: float
    irr_ok: bool
    payment_change: float
    payment_change_pct: float
    term_change: int
    capitalized_interest: float
    rejected_reason: Optional[str] = None
    requires_manual_review: bool = False

    def to_dict(self) -> Dict[str, float | str | bool | List[float]]:
        params_dict = asdict(self.params)
        params_payload: Dict[str, float | int] = {}
        for key, value in params_dict.items():
            if key == "reduction_factor":
                if value != 1.0:
                    params_payload[key] = value
                continue
            if isinstance(value, (int, float)):
                if value != 0:
                    params_payload[key] = value
            elif value not in (None, ""):
                params_payload[key] = value
        if not params_payload and self.params.months:
            params_payload["months"] = self.params.months
        if (
            self.params.reduction_factor != 1.0
            and "reduction_factor" not in params_payload
        ):
            params_payload["reduction_factor"] = self.params.reduction_factor

        return {
            "type": self.type,
            "params": params_payload,
            "new_payment": self.new_payment,
            "new_term": self.new_term,
            "cash_flows": self.cash_flows,
            "annual_irr": self.annual_irr,
            "irr_target": self.irr_target,
            "irr_ok": self.irr_ok,
            "payment_change": self.payment_change,
            "payment_change_pct": self.payment_change_pct,
            "term_change": self.term_change,
            "capitalized_interest": self.capitalized_interest,
            "rejected_reason": self.rejected_reason,
            "requires_manual_review": self.requires_manual_review,
        }


# ---------------------------------------------------------------------------
# Financial primitives (ported from TS implementation, adjusted to Python)
# ---------------------------------------------------------------------------


def calculate_irr(cash_flows: Sequence[float], guess: float = 0.01) -> Optional[float]:
    """Return monthly IRR for the cash flow series.

    Uses Newton-Raphson with bisection fallback to ensure robustness.
    """

    rate = guess
    tolerance = 1e-7
    max_iterations = 200

    def npv_and_derivative(r: float) -> Tuple[float, float]:
        npv = 0.0
        derivative = 0.0
        for t, cf in enumerate(cash_flows):
            denom = (1.0 + r) ** t
            npv += cf / denom
            if t > 0:
                derivative += -t * cf / (1.0 + r) ** (t + 1)
        return npv, derivative

    for _ in range(max_iterations):
        npv, derivative = npv_and_derivative(rate)
        if abs(npv) < tolerance:
            return rate
        if abs(derivative) < tolerance:
            break
        rate -= npv / derivative
        if rate < -0.99 or rate > 10:
            break

    # Fallback to bisection between -0.99 and 10
    lo, hi = -0.99, 10.0
    f_lo = sum(cf / (1.0 + lo) ** t for t, cf in enumerate(cash_flows))
    f_hi = sum(cf / (1.0 + hi) ** t for t, cf in enumerate(cash_flows))

    if f_lo * f_hi > 0:
        return None

    for _ in range(256):
        mid = (lo + hi) / 2.0
        f_mid = sum(cf / (1.0 + mid) ** t for t, cf in enumerate(cash_flows))
        if abs(f_mid) < tolerance:
            return mid
        if f_lo * f_mid < 0:
            hi = mid
            f_hi = f_mid
        else:
            lo = mid
            f_lo = f_mid
    return mid


def annuity_payment(principal: float, monthly_rate: float, term: int) -> float:
    if monthly_rate == 0:
        return principal / term
    numerator = monthly_rate * (1 + monthly_rate) ** term
    denominator = (1 + monthly_rate) ** term - 1
    return principal * numerator / denominator


def compute_implied_monthly_rate(principal: float, payment: float, term: int, *, max_rate: float = 5.0) -> float:
    if principal <= 0 or payment <= 0 or term <= 0:
        return 0.0

    zero_rate_payment = principal / term
    if abs(zero_rate_payment - payment) < 1e-9:
        return 0.0

    lo, hi = 0.0, max_rate
    for _ in range(200):
        mid = (lo + hi) / 2.0
        pay = annuity_payment(principal, mid, term)
        diff = pay - payment
        if abs(diff) < 1e-9:
            return mid
        if diff > 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2.0


def generate_cash_flows(principal: float, payments: Sequence[float]) -> List[float]:
    flows = [-principal]
    flows.extend(payments)
    return flows


def capitalize_interest(principal: float, monthly_rate: float, months: int) -> float:
    return principal * (1 + monthly_rate) ** months


def balance_after_payments(principal: float, monthly_payment: float, monthly_rate: float, months: int) -> float:
    balance = principal
    for _ in range(months):
        interest = balance * monthly_rate
        balance = balance + interest - monthly_payment
    return max(balance, 0.0)


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------


def get_tir_target(context: ProtectionContext, policy: ProtectionPolicy, *, principal: float, payment: float, term: int) -> float:
    monthly_rate = compute_implied_monthly_rate(principal, payment, term)
    annual_contract_rate = monthly_rate * 12
    market_floor = policy.tir_min_by_market.get(context.market.lower(), policy.tir_min_by_market.get("default", 0.0))
    return max(annual_contract_rate, market_floor)


def apply_policy_overrides(context: ProtectionContext, policy: ProtectionPolicy) -> Tuple[int, float]:
    max_deferral = policy.max_deferral_months
    max_reduction = policy.max_stepdown_reduction

    if context.has_consumption_gap or context.has_fault_alert:
        max_deferral = min(max_deferral, policy.soft_deferral_cap_with_gap)
        max_reduction = min(max_reduction, policy.soft_stepdown_cap_with_gap)

    if context.has_delinquency_flag or context.has_recent_promise_break:
        max_deferral = min(max_deferral, policy.soft_deferral_cap_with_gap)

    return max_deferral, max_reduction


def scenario_iterator(policy: ProtectionPolicy, context: ProtectionContext) -> Iterable[ScenarioParams]:
    max_deferral, max_reduction = apply_policy_overrides(context, policy)

    deferral_options = [m for m in (3, 6) if m <= max_deferral]
    stepdown_options = [m for m in (6,) if m <= policy.max_stepdown_months]
    stepdown_factor = max_reduction
    manual_plan = bool(context.requires_manual_review or (context.contract_status and context.contract_status.lower() != "active"))

    for months in deferral_options:
        yield ScenarioParams("DEFER", months=months)
    for months in stepdown_options:
        yield ScenarioParams("STEPDOWN", months=months, reduction_factor=stepdown_factor)
        deep_factor = min(0.3, stepdown_factor) if stepdown_factor > 0 else 0.0
        if manual_plan or deep_factor < stepdown_factor:
            yield ScenarioParams("STEPDOWN_DEEP", months=months, reduction_factor=deep_factor if deep_factor > 0 else stepdown_factor)
    for months in deferral_options:
        yield ScenarioParams("RECALENDAR", months=months)
    for extra in deferral_options or [1]:
        yield ScenarioParams("RECALENDAR_PARTIAL", deferral_months=extra)

    for deferral_months in deferral_options:
        for stepdown_months in stepdown_options:
            if deferral_months > 0 and stepdown_months > 0:
                yield ScenarioParams(
                    "DEFER_STEPDOWN",
                    deferral_months=deferral_months,
                    stepdown_months=stepdown_months,
                    reduction_factor=stepdown_factor,
                )

    if stepdown_factor < 1.0 and context.term_months > 0:
        yield ScenarioParams("BALLOON", reduction_factor=stepdown_factor)
        if manual_plan:
            yield ScenarioParams("BALLOON_STAGED", reduction_factor=stepdown_factor)



def generate_scenario(context: ProtectionContext, policy: ProtectionPolicy, params: ScenarioParams) -> Optional[ProtectionScenarioResult]:
    rate_target = get_tir_target(context, policy, principal=context.balance, payment=context.payment, term=context.term_months)
    tolerance = policy.tir_tolerance_bps / 10000

    monthly_rate = compute_implied_monthly_rate(context.balance, context.payment, context.term_months)
    manual_plan = bool(context.requires_manual_review or (context.contract_status and context.contract_status.lower() != "active"))
    result_params = params

    if params.type == "DEFER":
        deferral_months = params.months or params.deferral_months
        adjusted_balance = capitalize_interest(context.balance, monthly_rate, deferral_months)
        capitalized_interest = adjusted_balance - context.balance
        new_term = context.term_months - deferral_months
        if new_term <= 0:
            return None
        new_payment = annuity_payment(adjusted_balance, monthly_rate, new_term)
        payments = [0.0] * deferral_months + [new_payment] * new_term
        cash_flows = generate_cash_flows(context.balance, payments)
    elif params.type == "STEPDOWN":
        stepdown_months = params.months or params.stepdown_months
        reduction_factor = max(0.0, min(params.reduction_factor, 1.0))
        reduced_payment = context.payment * reduction_factor
        balance_after = balance_after_payments(context.balance, reduced_payment, monthly_rate, stepdown_months)
        remaining_term = context.term_months - stepdown_months
        if remaining_term <= 0:
            return None
        compensation_payment = annuity_payment(balance_after, monthly_rate, remaining_term)
        payments = [reduced_payment] * stepdown_months + [compensation_payment] * remaining_term
        adjusted_balance = context.balance
        capitalized_interest = 0.0
        new_payment = compensation_payment
        new_term = context.term_months
        cash_flows = generate_cash_flows(context.balance, payments)
    elif params.type == "STEPDOWN_DEEP":
        stepdown_months = params.months or params.stepdown_months
        reduction_factor = params.reduction_factor if params.reduction_factor else 0.3
        reduction_factor = max(0.0, min(reduction_factor, 1.0))
        reduced_payment = context.payment * reduction_factor
        balance_after = balance_after_payments(context.balance, reduced_payment, monthly_rate, stepdown_months)
        remaining_term = context.term_months - stepdown_months
        if remaining_term <= 0:
            return None
        compensation_payment = annuity_payment(balance_after, monthly_rate, remaining_term)
        payments = [reduced_payment] * stepdown_months + [compensation_payment] * remaining_term
        adjusted_balance = context.balance
        capitalized_interest = 0.0
        new_payment = compensation_payment
        new_term = context.term_months
        cash_flows = generate_cash_flows(context.balance, payments)
    elif params.type == "RECALENDAR":
        recalendar_months = params.months or params.deferral_months
        adjusted_balance = capitalize_interest(context.balance, monthly_rate, recalendar_months)
        capitalized_interest = adjusted_balance - context.balance
        new_term = context.term_months + recalendar_months
        new_payment = context.payment
        payments = [0.0] * recalendar_months + [context.payment] * context.term_months
        cash_flows = generate_cash_flows(adjusted_balance, payments)
    elif params.type == "RECALENDAR_PARTIAL":
        max_extra = policy.max_deferral_months
        start_extra = max(1, params.deferral_months or 1)
        target_monthly_rate = max(monthly_rate, rate_target / 12)
        selected_extra = None
        selected_payment = None
        selected_term = None
        selected_payments = None
        for extra in range(start_extra, max_extra + 1):
            prospective_term = context.term_months + extra
            prospective_payment = annuity_payment(context.balance, target_monthly_rate, prospective_term)
            prospective_payments = [prospective_payment] * prospective_term
            prospective_cash = generate_cash_flows(context.balance, prospective_payments)
            monthly_candidate = calculate_irr(prospective_cash)
            if monthly_candidate is None:
                continue
            annual_candidate = monthly_candidate * 12
            if (annual_candidate + tolerance) >= rate_target:
                selected_extra = extra
                selected_payment = prospective_payment
                selected_term = prospective_term
                selected_payments = prospective_payments
                break
        if selected_extra is None:
            selected_extra = max_extra
            selected_term = context.term_months + selected_extra
            selected_payment = annuity_payment(context.balance, target_monthly_rate, selected_term)
            selected_payments = [selected_payment] * selected_term
        capitalized_interest = 0.0
        new_payment = selected_payment
        new_term = selected_term
        payments = selected_payments
        cash_flows = generate_cash_flows(context.balance, payments)
        result_params = ScenarioParams("RECALENDAR_PARTIAL", deferral_months=selected_extra)
    elif params.type == "DEFER_STEPDOWN":
        deferral_months = params.deferral_months or params.months
        stepdown_months = params.stepdown_months or params.months
        if deferral_months <= 0 or stepdown_months <= 0:
            return None
        reduction_factor = max(0.0, min(params.reduction_factor, 1.0))
        adjusted_balance = capitalize_interest(context.balance, monthly_rate, deferral_months)
        capitalized_interest = adjusted_balance - context.balance
        remaining_term = context.term_months - stepdown_months
        if remaining_term <= 0:
            return None
        reduced_payment = context.payment * reduction_factor
        balance_after = balance_after_payments(adjusted_balance, reduced_payment, monthly_rate, stepdown_months)
        compensation_payment = annuity_payment(balance_after, monthly_rate, remaining_term)
        payments = ([0.0] * deferral_months + [reduced_payment] * stepdown_months + [compensation_payment] * remaining_term)
        new_payment = compensation_payment
        new_term = context.term_months + deferral_months
        cash_flows = generate_cash_flows(context.balance, payments)
    elif params.type == "BALLOON":
        reduction_factor = max(0.0, min(params.reduction_factor, 1.0))
        reduced_payment = context.payment * reduction_factor
        payments: List[float] = [reduced_payment] * context.term_months
        balance_after = balance_after_payments(context.balance, reduced_payment, monthly_rate, context.term_months)
        balloon_payment = 0.0
        if balance_after > 0:
            balloon_payment = capitalize_interest(balance_after, monthly_rate, 1)
            payments.append(balloon_payment)
        adjusted_balance = context.balance
        capitalized_interest = max(balloon_payment - balance_after, 0.0)
        new_payment = reduced_payment
        new_term = context.term_months + (1 if balloon_payment > 0 else 0)
        cash_flows = generate_cash_flows(context.balance, payments)
    elif params.type == "BALLOON_STAGED":
        reduction_factor = max(0.0, min(params.reduction_factor if params.reduction_factor else 0.5, 1.0))
        reduced_payment = context.payment * reduction_factor
        payments: List[float] = [reduced_payment] * context.term_months
        balance_after = balance_after_payments(context.balance, reduced_payment, monthly_rate, context.term_months)
        balloon_payment = 0.0
        if balance_after > 0:
            balloon_payment = capitalize_interest(balance_after, monthly_rate, 1)
            payments.append(balloon_payment * 0.6)
            payments.append(balloon_payment * 0.4)
        adjusted_balance = context.balance
        capitalized_interest = max(balloon_payment - balance_after, 0.0)
        new_payment = reduced_payment
        new_term = context.term_months + (2 if balance_after > 0 else 0)
        cash_flows = generate_cash_flows(context.balance, payments)
    else:
        return None

    monthly_irr = calculate_irr(cash_flows)
    if monthly_irr is None:
        irr_ok = False
        annual_irr = float("nan")
    else:
        annual_irr = monthly_irr * 12
        irr_ok = (annual_irr + tolerance) >= rate_target

    payment_change = new_payment - context.payment
    payment_change_pct = payment_change / context.payment * 100 if context.payment else 0.0
    term_change = new_term - context.term_months

    manual_review = manual_plan or (params.type in {"STEPDOWN_DEEP", "BALLOON_STAGED"})

    return ProtectionScenarioResult(
        type=result_params.type,
        params=result_params,
        new_payment=new_payment,
        new_term=new_term,
        cash_flows=list(cash_flows),
        annual_irr=annual_irr,
        irr_target=rate_target,
        irr_ok=irr_ok,
        payment_change=payment_change,
        payment_change_pct=payment_change_pct,
        term_change=term_change,
        capitalized_interest=capitalized_interest,
        rejected_reason=None if irr_ok else "IRR below minimum",
        requires_manual_review=manual_review,
    )



def evaluate_protection_scenarios(context: ProtectionContext, policy: ProtectionPolicy, *, scenario_plan: Optional[Iterable[ScenarioParams]] = None) -> List[ProtectionScenarioResult]:
    if context.protections_allowed is not None and context.protections_used >= context.protections_allowed:
        return []
    if context.restructures_allowed is not None and context.restructures_used >= context.restructures_allowed:
        return []
    if context.restructures_used >= policy.max_restructures:
        return []

    plan = scenario_plan or scenario_iterator(policy, context)
    results: List[ProtectionScenarioResult] = []
    for params in plan:
        result = generate_scenario(context, policy, params)
        if result is not None:
            results.append(result)
    return results


def select_viable_scenarios(results: Iterable[ProtectionScenarioResult]) -> List[ProtectionScenarioResult]:
    viable = [r for r in results if r.irr_ok]
    return sorted(
        viable,
        key=lambda r: (
            r.requires_manual_review,
            abs(r.payment_change),
            r.term_change,
            -r.annual_irr,
        ),
    )


def load_policy_from_config(path: Path | None = None) -> ProtectionPolicy:
    config_path = path or DEFAULT_CONFIG_PATH
    default_policy = ProtectionPolicy(
        tir_min_by_market={
            "aguascalientes": 0.255,
            "edomex": 0.299,
            "default": 0.255,
        },
        tir_tolerance_bps=25,
        max_restructures=2,
        max_deferral_months=6,
        max_stepdown_months=6,
        max_stepdown_reduction=0.5,
        soft_deferral_cap_with_gap=3,
        soft_stepdown_cap_with_gap=0.25,
    )

    try:
        with config_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except FileNotFoundError:
        return default_policy
    except Exception:
        return default_policy

    markets = data.get("markets", {}) or {}
    policy_cfg = data.get("policy", {}) or {}

    tir_map = {k.lower(): float(v.get("tir_min", default_policy.tir_min_by_market.get(k.lower(), 0.255))) for k, v in markets.items()}
    for key, val in default_policy.tir_min_by_market.items():
        tir_map.setdefault(key, val)

    def _get(name: str, fallback: float | int) -> float | int:
        value = policy_cfg.get(name, fallback)
        try:
            return type(fallback)(value)
        except Exception:
            return fallback

    return ProtectionPolicy(
        tir_min_by_market=tir_map,
        tir_tolerance_bps=int(_get("tir_tolerance_bps", default_policy.tir_tolerance_bps)),
        max_restructures=int(_get("max_restructures", default_policy.max_restructures)),
        max_deferral_months=int(_get("max_deferral_months", default_policy.max_deferral_months)),
        max_stepdown_months=int(_get("max_stepdown_months", default_policy.max_stepdown_months)),
        max_stepdown_reduction=float(_get("max_stepdown_reduction", default_policy.max_stepdown_reduction)),
        soft_deferral_cap_with_gap=int(_get("soft_deferral_cap_with_gap", default_policy.soft_deferral_cap_with_gap)),
        soft_stepdown_cap_with_gap=float(_get("soft_stepdown_cap_with_gap", default_policy.soft_stepdown_cap_with_gap)),
    )


DEFAULT_POLICY = load_policy_from_config()


def get_default_policy(*, reload: bool = False) -> ProtectionPolicy:
    """Return the cached policy, optionally forcing a reload from disk."""

    global DEFAULT_POLICY
    if reload:
        DEFAULT_POLICY = load_policy_from_config()
    return DEFAULT_POLICY

__all__ = [
    "ProtectionPolicy",
    "ProtectionContext",
    "ScenarioParams",
    "ProtectionScenarioResult",
    "DEFAULT_POLICY",
    "get_default_policy",
    "load_policy_from_config",
    "evaluate_protection_scenarios",
    "select_viable_scenarios",
]
