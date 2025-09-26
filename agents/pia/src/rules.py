"""Reglas y lógica de decisión para el agente PIA."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, UTC
from typing import Dict, Optional

from .config import DEFAULT_CONFIG


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
    thresholds = DEFAULT_CONFIG.risk_thresholds
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
    triggers = DEFAULT_CONFIG.triggers
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
    return delta.days >= DEFAULT_CONFIG.triggers.cooldown_days_essential


def calculate_collected(payload: dict) -> float:
    expected = float(payload.get("expected_payment", 0))
    arrears = float(payload.get("arrears_amount", 0))
    bank = float(payload.get("bank_transfer", 0))
    gnv = float(payload.get("gnv_credit_30d", 0))
    synthetic = expected - arrears
    observed = bank + gnv
    return max(synthetic, observed, 0.0)


def detect_consumption_gap(payload: dict, expected_payment: float, observed_payment: float) -> bool:
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
        baseline_litros = float(payload.get("avg_daily_litros", 0)) * 30
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
    intents = DEFAULT_CONFIG.intents
    if value in intents.balance_inquiry:
        return "balance_inquiry"
    if value in intents.advance_payment:
        return "advance_payment"
    if value in intents.payment_promise:
        return "payment_promise"
    if value in intents.document_support:
        return "document_support"
    return value


def decide_action(payload: dict) -> PIADecision:
    placa = payload.get("placa", "UNKNOWN")
    risk_score = float(payload.get("risk_score", 0))
    risk_band = categorize_risk(risk_score)
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
    amounts = DEFAULT_CONFIG.amounts

    if intent == "balance_inquiry":
        return PIADecision(
            placa=placa,
            risk_band=risk_band,
            action="provide_balance",
            reason="Consulta de saldo",
            template="PIA_RECORDATORIO",
            details=details,
        )

    if intent == "advance_payment":
        return PIADecision(
            placa=placa,
            risk_band=risk_band,
            action="prepare_advance_payment",
            reason="Cliente desea pagar anticipadamente",
            template="PIA_RECORDATORIO",
            details=details,
        )

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
        return PIADecision(
            placa=placa,
            risk_band=risk_band,
            action="record_payment_promise",
            reason="Cliente promete un pago",
            template="PIA_PROMESA",
            details=details,
        )

    if intent == "document_support":
        document_type = payload.get("document_type") or payload.get("document_requested")
        if document_type:
            details["document_type"] = str(document_type)
        return PIADecision(
            placa=placa,
            risk_band=risk_band,
            action="provide_documents",
            reason="Cliente solicita documentación",
            template="PIA_DOCUMENTOS",
            details=details,
        )

    if detect_consumption_gap(payload, expected_payment, observed_payment):
        details["hase_consumption_gap_flag"] = int(payload.get("hase_consumption_gap_flag", 0))
        details["telemetry_ok"] = bool(int(payload.get("hase_telemetry_ok_flag", 0)))
        return PIADecision(
            placa=placa,
            risk_band=risk_band,
            action="investigate_consumption",
            reason="Consumo GNV bajo con operación activa",
            template="PIA_CONSUMO",
            details=details,
        )

    if detect_fault_alert(payload):
        details["hase_fault_alert_flag"] = int(payload.get("hase_fault_alert_flag", 0))
        return PIADecision(
            placa=placa,
            risk_band=risk_band,
            action="escalate_fault_check",
            reason="Telemetría reporta fallas críticas",
            template="PIA_FALLA",
            details=details,
        )

    if arrears <= amounts.overpayment_margin and observed_payment >= expected_payment:
        return PIADecision(
            placa=placa,
            risk_band=risk_band,
            action="acknowledge_prepayment",
            reason="Pago anticipado detectado",
            template="PIA_RECORDATORIO",
            details=details,
        )

    if should_offer_protection(payload) and respect_cooldown(last_protection):
        return PIADecision(
            placa=placa,
            risk_band=risk_band,
            action="offer_protection",
            scenario=payload.get("suggested_scenario", "restructure-light"),
            template="PIA_OPCIONES",
            reason="Cobertura baja y telemetría crítica",
            details=details,
        )

    if risk_band in {"alto", "muy_alto"} or arrears > amounts.arrears_tolerance:
        return PIADecision(
            placa=placa,
            risk_band=risk_band,
            action="payment_reminder",
            template="PIA_RECORDATORIO",
            reason="Riesgo elevado o adeudo detectado",
            details=details,
        )

    return PIADecision(
        placa=placa,
        risk_band=risk_band,
        action="check_in",
        template="PIA_RECORDATORIO",
        reason="Seguimiento preventivo",
        details=details,
    )
