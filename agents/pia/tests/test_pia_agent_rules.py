import csv
import os
from datetime import datetime, timezone, timedelta

from agents.pia.src import config as pia_config
from agents.pia.src.outcomes import record_outcome
from agents.pia.src.rules import PIADecision, categorize_risk, decide_action


def test_categorize_risk_thresholds():
    assert categorize_risk(0.5) == "bajo"
    assert categorize_risk(0.7) == "medio"
    assert categorize_risk(0.8) == "alto"
    assert categorize_risk(0.9) == "muy_alto"


def _build_payload(**overrides):
    base = {
        "placa": "TEST",
        "risk_score": 0.72,
        "coverage_ratio_14d": 0.9,
        "coverage_ratio_30d": 0.8,
        "downtime_hours_30d": 0,
        "arrears_amount": 0,
        "expected_payment": 11000.0,
        "bank_transfer": 0.0,
        "gnv_credit_30d": 0.0,
        "avg_daily_litros": 0.0,
        "avg_daily_recaudo": 0.0,
        "avg_30d_litros": 0.0,
        "avg_30d_recaudo": 0.0,
        "distance_km_30d": 350.0,
        "engine_hours_30d": 90.0,
        "fault_events_30d": 0.0,
        "hase_consumption_gap_flag": 0,
        "hase_fault_alert_flag": 0,
        "hase_telemetry_ok_flag": 0,
        "last_protection_at": None,
    }
    base.update(overrides)
    return base


def test_decide_action_offer_protection():
    payload = _build_payload(
        coverage_ratio_14d=0.5,
        coverage_ratio_30d=0.6,
        downtime_hours_30d=200,
        suggested_scenario="restructure-full",
    )
    decision = decide_action(payload)
    assert decision.action == "offer_protection"
    assert decision.template == "PIA_OPCIONES"
    assert decision.scenario == "restructure-full"
    assert decision.details["expected_payment"] == 11000.0


def test_decide_action_payment_reminder():
    payload = _build_payload(risk_score=0.85, arrears_amount=1500)
    decision = decide_action(payload)
    assert decision.action == "payment_reminder"
    assert decision.template == "PIA_RECORDATORIO"
    assert decision.details["arrears_amount"] == 1500


def test_decide_action_check_in():
    payload = _build_payload(risk_score=0.5)
    decision = decide_action(payload)
    assert decision.action == "check_in"
    assert isinstance(decision, PIADecision)


def test_decide_action_respects_cooldown():
    payload = _build_payload(
        coverage_ratio_14d=0.4,
        coverage_ratio_30d=0.55,
        downtime_hours_30d=180,
        last_protection_at="2025-09-20T00:00:00",
    )
    decision = decide_action(payload)
    assert decision.action == "check_in"


def test_decide_action_payment_due_without_high_risk():
    payload = _build_payload(risk_score=0.55, arrears_amount=800)
    decision = decide_action(payload)
    assert decision.action == "payment_reminder"
    assert decision.reason == "Riesgo elevado o adeudo detectado"


def test_decide_action_protection_after_cooldown():
    payload = _build_payload(
        coverage_ratio_14d=0.5,
        coverage_ratio_30d=0.6,
        downtime_hours_30d=190,
        last_protection_at="2024-01-01T00:00:00",
        suggested_scenario="restructure-light",
    )
    decision = decide_action(payload)
    assert decision.action == "offer_protection"
    assert decision.scenario == "restructure-light"


def test_decide_action_acknowledge_prepayment():
    payload = _build_payload(arrears_amount=-300, bank_transfer=6000, gnv_credit_30d=6000)
    decision = decide_action(payload)
    assert decision.action == "acknowledge_prepayment"
    assert decision.details["collected_amount"] > decision.details["expected_payment"]


def test_decide_action_balance_query_intent():
    payload = _build_payload(intent="consulta_saldo", arrears_amount=25, bank_transfer=4000, gnv_credit_30d=7000)
    decision = decide_action(payload)
    assert decision.action == "provide_balance"
    assert decision.details["collected_amount"] >= 11000 - 25


def test_decide_action_prepare_advance_payment():
    payload = _build_payload(intent="pago_anticipado", arrears_amount=0)
    decision = decide_action(payload)
    assert decision.action == "prepare_advance_payment"
    assert decision.reason == "Cliente desea pagar anticipadamente"


def test_decide_action_no_ack_without_observed_payment():
    payload = _build_payload(arrears_amount=-200, bank_transfer=0, gnv_credit_30d=0)
    decision = decide_action(payload)
    assert decision.action == "check_in"


def test_decide_action_records_payment_promise():
    payload = _build_payload(
        intent="promesa_pago",
        promise_date="2025-10-15",
        promised_amount=4200,
        arrears_amount=500,
    )
    decision = decide_action(payload)
    assert decision.action == "record_payment_promise"
    assert decision.template == "PIA_PROMESA"
    assert decision.details["promise_date"] == "2025-10-15"
    assert decision.details["promised_amount"] == 4200.0


def test_decide_action_handles_document_support():
    payload = _build_payload(
        intent="soporte_documental",
        document_type="estado_de_cuenta",
        arrears_amount=25,
    )
    decision = decide_action(payload)
    assert decision.action == "provide_documents"
    assert decision.template == "PIA_DOCUMENTOS"
    assert decision.details["document_type"] == "estado_de_cuenta"


def test_decide_action_consumption_gap_flag_direct():
    payload = _build_payload(
        hase_consumption_gap_flag=1,
        hase_telemetry_ok_flag=1,
        distance_km_30d=1200,
        engine_hours_30d=240,
        risk_score=0.68,
        avg_30d_litros=1800,
    )
    decision = decide_action(payload)
    assert decision.action == "investigate_consumption"
    assert decision.template == "PIA_CONSUMO"


def test_decide_action_consumption_gap_inferred():
    payload = _build_payload(
        hase_telemetry_ok_flag=1,
        distance_km_30d=1300,
        engine_hours_30d=250,
        gnv_credit_30d=50,
        avg_30d_litros=1800,
        observed_payment=0,
    )
    payload.pop("observed_payment", None)
    decision = decide_action(payload)
    assert decision.action == "investigate_consumption"


def test_decide_action_gnv_min_floor_triggers_gap():
    payload = _build_payload(
        hase_telemetry_ok_flag=1,
        distance_km_30d=800,
        engine_hours_30d=200,
        gnv_credit_30d=100,  # Bajo recaudo
        avg_30d_litros=80,    # 80 litros => sobreprecio $1.25 < $5
        observed_payment=0,
    )
    payload.pop("observed_payment", None)
    decision = decide_action(payload)
    assert decision.action == "investigate_consumption"
    assert decision.details.get("gnv_min_floor_breached")


def test_decide_action_fault_alert():
    payload = _build_payload(
        hase_fault_alert_flag=1,
        fault_events_30d=3,
        arrears_amount=400,
    )
    decision = decide_action(payload)
    assert decision.action == "escalate_fault_check"
    assert decision.template == "PIA_FALLA"


def test_decide_action_memory_suppresses_repeat(tmp_path):
    payload = _build_payload(risk_score=0.9, arrears_amount=500)
    first_decision = decide_action(payload)
    assert first_decision.action == "payment_reminder"

    log_path = tmp_path / "pia_outcomes.csv"
    record_outcome(first_decision, "payment_reminder_sent", log_path=log_path)

    os.environ["PIA_OUTCOMES_LOG_PATH"] = str(log_path)
    try:
        repeat_decision = decide_action(payload)
    finally:
        os.environ.pop("PIA_OUTCOMES_LOG_PATH", None)

    assert repeat_decision.action == pia_config.get_config().memory.escalate_action
    memory_info = repeat_decision.details.get("memory_repeat") or {}
    assert memory_info.get("suppressed") is True


def test_decide_action_protection_followup(tmp_path):
    base_payload = _build_payload(
        coverage_ratio_14d=0.4,
        coverage_ratio_30d=0.5,
        downtime_hours_30d=190,
        suggested_scenario="restructure-light",
        arrears_amount=600,
    )
    initial_decision = decide_action(base_payload)
    assert initial_decision.action == "offer_protection"

    log_path = tmp_path / "pia_followup.csv"
    record_outcome(initial_decision, "proposed_protection", log_path=log_path)

    follow_cfg = pia_config.get_config().followup
    past_ts = datetime.now(timezone.utc) - timedelta(hours=follow_cfg.protection_followup_hours + 1)
    with log_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames or list(rows[0].keys())
    rows[0]["timestamp"] = past_ts.isoformat()
    with log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    os.environ["PIA_OUTCOMES_LOG_PATH"] = str(log_path)
    try:
        followup_decision = decide_action(base_payload)
    finally:
        os.environ.pop("PIA_OUTCOMES_LOG_PATH", None)

    assert followup_decision.action == "offer_protection"
    assert followup_decision.template == "PIA_SEGUIMIENTO"
    assert followup_decision.details.get("protection_followup") is True
    assert followup_decision.details.get("force_repeat") is True
