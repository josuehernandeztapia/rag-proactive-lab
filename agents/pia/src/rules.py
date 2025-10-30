"""Reglas y lógica de decisión para el agente PIA."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, UTC, timedelta
from typing import Dict, Iterable, Optional

from . import config as pia_config
from .cobranza import CobranzaCase, preparar_cobranza_payload


@dataclass
class PIADecision:
    placa: str
    risk_band: str
    action: str
    reason: str
    scenario: Optional[str] = None
    template: str = "PIA_RECORDATORIO"
    details: Dict[str, float | str] = field(default_factory=dict)


def categorize_risk(score: float) -> str:
    config = pia_config.get_config()
    thresholds = config.risk_thresholds
    if score >= thresholds.high:
        return "muy_alto"
    if score >= thresholds.medium:
        return "alto"
    if score >= thresholds.low:
        return "medio"
    return "bajo"


def should_offer_protection(payload: dict) -> bool:
    coverage14 = float(payload.get("coverage_ratio_14d", 1))
    coverage30 = float(payload.get("coverage_ratio_30d", 1))
    downtime = float(payload.get("downtime_hours_30d", 0))
    triggers = pia_config.get_config().triggers
    if coverage14 < triggers.coverage_low_14d and coverage30 < triggers.coverage_low_30d:
        return True
    if downtime > triggers.downtime_high_hours_14d:
        return True
    return False


def respect_cooldown(last_protection_at: Optional[str]) -> bool:
    if not last_protection_at:
        return True
    try:
        last_dt = datetime.fromisoformat(str(last_protection_at))
    except ValueError:
        return True
    delta = datetime.now(UTC) - last_dt.astimezone(UTC)
    return delta.days >= pia_config.get_config().triggers.cooldown_days_essential


def calculate_collected(payload: dict) -> float:
    expected = float(payload.get("expected_payment", 0))
    arrears = float(payload.get("arrears_amount", 0))
    bank = float(payload.get("bank_transfer", 0))
    gnv = float(payload.get("gnv_credit_30d", 0))
    synthetic = expected - arrears
    observed = bank + gnv
    return max(synthetic, observed, 0.0)


def detect_consumption_gap(
    payload: dict,
    expected_payment: float,
    observed_payment: float,
    *,
    gnv_min_breached: bool = False,
) -> bool:
    if gnv_min_breached:
        return True
    if int(payload.get("hase_consumption_gap_flag", 0)):
        return True
    telemetry_ok = bool(int(payload.get("hase_telemetry_ok_flag", 0)))
    if expected_payment <= 0:
        return False
    consumption = float(payload.get("gnv_credit_30d", 0))
    distance = float(payload.get("distance_km_30d", 0))
    engine_hours = float(payload.get("engine_hours_30d", 0))
    low_collection = observed_payment <= expected_payment * 0.2
    baseline_litros = float(payload.get("avg_30d_litros", 0))
    if baseline_litros <= 0:
        config = pia_config.get_config()
        fallback_days = max(config.gnv.liters_fallback_days, 1)
        baseline_litros = float(payload.get("avg_daily_litros", 0)) * fallback_days
    if baseline_litros > 0:
        low_consumption = consumption <= baseline_litros * 0.3
    else:
        low_consumption = consumption <= max(300.0, expected_payment * 0.15)
    high_activity = distance >= 400 or engine_hours >= 120
    return telemetry_ok and low_collection and low_consumption and high_activity


def detect_fault_alert(payload: dict) -> bool:
    if int(payload.get("hase_fault_alert_flag", 0)):
        return True
    return float(payload.get("fault_events_30d", 0)) >= 2


def normalize_intent(raw_intent: Optional[str]) -> str:
    if not raw_intent:
        return ""
    value = str(raw_intent).strip().lower()
    intents = pia_config.get_config().intents
    if value in intents.balance_inquiry:
        return "balance_inquiry"
    if value in intents.advance_payment:
        return "advance_payment"
    if value in intents.payment_promise:
        return "payment_promise"
    if value in intents.document_support:
        return "document_support"
    return value


def _compute_gnv_metrics(payload: dict) -> tuple[float, bool]:
    config = pia_config.get_config()
    gnv_credit = float(payload.get("gnv_credit_30d", 0))
    liters = float(payload.get("avg_30d_litros", 0))
    if liters <= config.gnv.epsilon_liters:
        daily = float(payload.get("avg_daily_litros", 0))
        fallback_days = max(config.gnv.liters_fallback_days, 1)
        liters = daily * fallback_days
    if liters <= config.gnv.epsilon_liters:
        return 0.0, False
    overprice = gnv_credit / max(liters, config.gnv.epsilon_liters)
    breached = overprice < config.gnv.min_overprice_per_liter
    return overprice, breached


def _to_utc(value: Optional[object]) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.astimezone(UTC)
    if hasattr(value, "to_pydatetime"):
        try:
            dt = value.to_pydatetime()
            return dt.astimezone(UTC)
        except Exception:
            return None
    if isinstance(value, str):
        raw = value.strip()
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        try:
            return datetime.fromisoformat(raw).astimezone(UTC)
        except ValueError:
            return None
    return None


def _fetch_recent_outcome(
    placa: str,
    *,
    actions: Optional[Iterable[str]] = None,
    outcomes: Optional[Iterable[str]] = None,
    within_hours: Optional[int] = None,
) -> Optional[Dict[str, object]]:
    try:
        from .outcomes import fetch_recent_outcomes
    except Exception:
        return None
    rows = fetch_recent_outcomes(
        placa,
        within_hours=within_hours,
        limit=1,
        actions=actions,
        outcomes=outcomes,
    )
    return rows[0] if rows else None


def _build_cobranza_case(payload: dict, arrears: float) -> CobranzaCase:
    dias_keys = ["dias_mora", "days_past_due", "dias_en_mora"]
    dias_val = 0
    for key in dias_keys:
        value = payload.get(key)
        if value is not None:
            try:
                dias_val = int(float(value))
                break
            except (TypeError, ValueError):
                continue
    saldo = payload.get("saldo_pendiente")
    if saldo is None:
        saldo = payload.get("balance")
    try:
        saldo_val = float(saldo) if saldo is not None else 0.0
    except (TypeError, ValueError):
        saldo_val = 0.0
    last_response_raw = payload.get("last_client_response_at") or payload.get("ultima_respuesta")
    last_response = None
    if isinstance(last_response_raw, datetime):
        last_response = last_response_raw
    elif isinstance(last_response_raw, str):
        try:
            last_response = datetime.fromisoformat(last_response_raw.replace("Z", "+00:00"))
        except ValueError:
            last_response = None
    context = payload.get("client_context")
    if not isinstance(context, dict):
        context = {}
    return CobranzaCase(
        placa=str(payload.get("placa", "")) or "SIN-PLACA",
        dias_mora=max(dias_val, 0),
        monto_vencido=max(float(arrears), 0.0),
        saldo_pendiente=saldo_val,
        intentos_contacto=int(payload.get("cobranza_intentos", 0) or 0),
        ultima_respuesta=last_response,
        protection_offered=bool(payload.get("cobranza_protection_offered") or payload.get("protection_offered")),
        escalated_to_advisor=bool(payload.get("cobranza_escalated") or payload.get("escalated_to_advisor")),
        client_context=context,
    )


def _enrich_with_cobranza(decision: PIADecision, payload: dict, arrears: float) -> PIADecision:
    actionable = {"payment_reminder", "check_in", "investigate_consumption", "offer_protection"}
    if decision.action not in actionable:
        return decision
    if arrears <= 0 and not any(payload.get(k) for k in ("dias_mora", "days_past_due", "dias_en_mora")):
        return decision
    case = _build_cobranza_case(payload, arrears)
    try:
        cobranza_info = preparar_cobranza_payload(case)
    except Exception:
        cobranza_info = None
    if cobranza_info:
        decision.details.setdefault("cobranza", cobranza_info)
    return decision


def _finalize_decision(decision: PIADecision, payload: dict, arrears: float) -> PIADecision:
    enriched = _enrich_with_cobranza(decision, payload, arrears)
    return _apply_memory(enriched, payload)


def _maybe_followup_protection(
    payload: dict,
    base_details: dict,
    risk_band: str,
    arrears: float,
    observed_payment: float,
) -> Optional[PIADecision]:
    config = pia_config.get_config()
    follow_cfg = config.followup
    if follow_cfg.protection_followup_hours <= 0:
        return None
    placa = str(payload.get("placa") or "").strip()
    if not placa:
        return None
    pending = _fetch_recent_outcome(
        placa,
        outcomes={"proposed_protection"},
    )
    if not pending:
        return None
    pending_ts = _to_utc(pending.get("timestamp"))
    if not pending_ts:
        return None
    # Skip if there is a more recent completion/decline
    completion = _fetch_recent_outcome(
        placa,
        outcomes={"protection_executed", "protection_declined"},
    )
    if completion:
        completion_ts = _to_utc(completion.get("timestamp"))
        if completion_ts and completion_ts >= pending_ts:
            return None
    age = datetime.now(UTC) - pending_ts
    if age < timedelta(hours=follow_cfg.protection_followup_hours):
        return None
    amounts = config.amounts
    expected_payment = float(payload.get("expected_payment", 0))
    still_due = (
        arrears > amounts.arrears_tolerance
        or observed_payment + amounts.arrears_tolerance < expected_payment
    )
    if not still_due:
        return None
    details = dict(base_details)
    details["protection_followup"] = True
    details["force_repeat"] = True
    details["pending_protection_generated_at"] = pending_ts.isoformat()
    details["pending_protection_outcome"] = pending.get("outcome")
    scenario_hint = pending.get("scenario") or (pending.get("details_json") or {}).get("scenario") if isinstance(pending.get("details_json"), dict) else None
    return PIADecision(
        placa=placa,
        risk_band=risk_band,
        action="offer_protection",
        reason="Protección pendiente sin ejecución",
        template="PIA_SEGUIMIENTO",
        scenario=payload.get("suggested_scenario") or scenario_hint,
        details=details,
    )


def _apply_memory(decision: PIADecision, payload: dict) -> PIADecision:
    config = pia_config.get_config()
    memory_cfg = config.memory
    if not memory_cfg.enabled:
        return decision
    if decision.details.get("force_repeat"):
        return decision
    placa = str(decision.placa or "").strip()
    if not placa:
        return decision
    recent = _fetch_recent_outcome(
        placa,
        actions={decision.action},
        within_hours=memory_cfg.repeat_cooldown_hours,
    )
    if not recent:
        return decision
    last_ts = _to_utc(recent.get("timestamp"))
    memory_info = {
        "action": recent.get("action"),
        "last_timestamp": last_ts.isoformat() if last_ts else None,
    }
    if decision.action == memory_cfg.escalate_action:
        decision.details["memory_repeat"] = memory_info
        return decision
    if decision.action in memory_cfg.annotate_actions:
        decision.details["memory_repeat"] = memory_info
        return decision
    if decision.action in memory_cfg.suppress_actions:
        decision.details["memory_repeat"] = {**memory_info, "suppressed": True}
        decision.action = memory_cfg.escalate_action
        decision.template = memory_cfg.escalate_template
        decision.reason = memory_cfg.escalate_reason
        return decision
    decision.details["memory_repeat"] = memory_info
    return decision


def decide_action(payload: dict) -> PIADecision:
    placa = payload.get("placa", "UNKNOWN")
    risk_score = float(payload.get("risk_score", 0))
    risk_band = categorize_risk(risk_score)
    config = pia_config.get_config()
    arrears = float(payload.get("arrears_amount", 0))
    last_protection = payload.get("last_protection_at")
    intent = normalize_intent(payload.get("intent"))
    expected_payment = float(payload.get("expected_payment", 0))
    bank_transfer = float(payload.get("bank_transfer", 0))
    gnv_credit = float(payload.get("gnv_credit_30d", 0))
    observed_payment = bank_transfer + gnv_credit
    collected_amount = calculate_collected(payload)
    distance_km = float(payload.get("distance_km_30d", 0))
    engine_hours = float(payload.get("engine_hours_30d", 0))
    fault_events = float(payload.get("fault_events_30d", 0))
    avg_30d_litros = float(payload.get("avg_30d_litros", 0))
    avg_30d_recaudo = float(payload.get("avg_30d_recaudo", 0))
    details = {
        "expected_payment": expected_payment,
        "collected_amount": collected_amount,
        "arrears_amount": arrears,
        "observed_payment": observed_payment,
        "gnv_credit_30d": gnv_credit,
        "distance_km_30d": distance_km,
        "engine_hours_30d": engine_hours,
        "fault_events_30d": fault_events,
        "avg_30d_litros": avg_30d_litros,
        "avg_30d_recaudo": avg_30d_recaudo,
    }
    gnv_overprice, gnv_min_breached = _compute_gnv_metrics(payload)
    details["gnv_overprice_per_liter"] = gnv_overprice
    if gnv_min_breached:
        details["gnv_min_floor_breached"] = True
    amounts = config.amounts

    if intent == "balance_inquiry":
        decision = PIADecision(
            placa=placa,
            risk_band=risk_band,
            action="provide_balance",
            reason="Consulta de saldo",
            template="PIA_RECORDATORIO",
            details=details,
        )
        return _finalize_decision(decision, payload, arrears)

    if intent == "advance_payment":
        decision = PIADecision(
            placa=placa,
            risk_band=risk_band,
            action="prepare_advance_payment",
            reason="Cliente desea pagar anticipadamente",
            template="PIA_RECORDATORIO",
            details=details,
        )
        return _finalize_decision(decision, payload, arrears)

    if intent == "payment_promise":
        promise_date = payload.get("promise_date")
        promised_amount = payload.get("promised_amount")
        if promise_date:
            details["promise_date"] = str(promise_date)
        if promised_amount is not None:
            try:
                details["promised_amount"] = float(promised_amount)
            except (TypeError, ValueError):  # pragma: no cover - defensive
                details["promised_amount"] = promised_amount
        decision = PIADecision(
            placa=placa,
            risk_band=risk_band,
            action="record_payment_promise",
            reason="Cliente promete un pago",
            template="PIA_PROMESA",
            details=details,
        )
        return _finalize_decision(decision, payload, arrears)

    if intent == "document_support":
        document_type = payload.get("document_type") or payload.get("document_requested")
        if document_type:
            details["document_type"] = str(document_type)
        decision = PIADecision(
            placa=placa,
            risk_band=risk_band,
            action="provide_documents",
            reason="Cliente solicita documentación",
            template="PIA_DOCUMENTOS",
            details=details,
        )
        return _finalize_decision(decision, payload, arrears)

    followup_decision = _maybe_followup_protection(
        payload,
        details,
        risk_band,
        arrears,
        observed_payment,
    )
    if followup_decision:
        return _finalize_decision(followup_decision, payload, arrears)

    if detect_consumption_gap(
        payload,
        expected_payment,
        observed_payment,
        gnv_min_breached=gnv_min_breached,
    ):
        details["hase_consumption_gap_flag"] = int(payload.get("hase_consumption_gap_flag", 0))
        details["telemetry_ok"] = bool(int(payload.get("hase_telemetry_ok_flag", 0)))
        decision = PIADecision(
            placa=placa,
            risk_band=risk_band,
            action="investigate_consumption",
            reason="Consumo GNV bajo con operación activa",
            template="PIA_CONSUMO",
            details=details,
        )
        return _finalize_decision(decision, payload, arrears)

    if detect_fault_alert(payload):
        details["hase_fault_alert_flag"] = int(payload.get("hase_fault_alert_flag", 0))
        decision = PIADecision(
            placa=placa,
            risk_band=risk_band,
            action="escalate_fault_check",
            reason="Telemetría reporta fallas críticas",
            template="PIA_FALLA",
            details=details,
        )
        return _finalize_decision(decision, payload, arrears)

    if arrears <= amounts.overpayment_margin and observed_payment >= expected_payment:
        decision = PIADecision(
            placa=placa,
            risk_band=risk_band,
            action="acknowledge_prepayment",
            reason="Pago anticipado detectado",
            template="PIA_RECORDATORIO",
            details=details,
        )
        return _finalize_decision(decision, payload, arrears)

    if should_offer_protection(payload) and respect_cooldown(last_protection):
        decision = PIADecision(
            placa=placa,
            risk_band=risk_band,
            action="offer_protection",
            scenario=payload.get("suggested_scenario", "restructure-light"),
            template="PIA_OPCIONES",
            reason="Cobertura baja y telemetría crítica",
            details=details,
        )
        return _finalize_decision(decision, payload, arrears)

    if risk_band in {"alto", "muy_alto"} or arrears > amounts.arrears_tolerance:
        decision = PIADecision(
            placa=placa,
            risk_band=risk_band,
            action="payment_reminder",
            template="PIA_RECORDATORIO",
            reason="Riesgo elevado o adeudo detectado",
            details=details,
        )
        return _finalize_decision(decision, payload, arrears)

    decision = PIADecision(
        placa=placa,
        risk_band=risk_band,
        action="check_in",
        template="PIA_RECORDATORIO",
        reason="Seguimiento preventivo",
        details=details,
    )
    return _finalize_decision(decision, payload, arrears)
