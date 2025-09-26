from dataclasses import asdict

from agents.pia.src.outcomes import aggregate_outcomes, load_outcomes, record_outcome
from agents.pia.src.rules import PIADecision


def _sample_decision(**overrides):
    base = PIADecision(
        placa="TEST-001",
        risk_band="medio",
        action="investigate_consumption",
        reason="Consumo GNV bajo con operación activa",
        scenario="consumption_gap",
        template="PIA_CONSUMO",
        details={
            "expected_payment": 11000.0,
            "collected_amount": 2000.0,
            "arrears_amount": 9000.0,
            "plaza": "EDOMEX",
        },
    )
    if overrides:
        payload = asdict(base)
        payload.update(overrides)
        return PIADecision(**payload)
    return base


def test_record_and_load_outcome(tmp_path):
    log_path = tmp_path / "outcomes.csv"
    decision = _sample_decision()
    record_outcome(
        decision,
        "resolved",
        plaza="EDOMEX",
        metadata={
            "channel": "whatsapp",
            "protection_plan": {
                "plan_type": "proteccion_total",
                "protections_used": 1,
                "protections_allowed": 2,
                "status": "active",
                "valid_until": "2026-12-31",
                "reset_cycle_days": 365,
                "requires_manual_review": True,
            },
        },
        log_path=log_path,
    )
    df = load_outcomes(log_path)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["outcome"] == "resolved"
    assert row["plaza"] == "EDOMEX"
    assert row["metadata_json"] == {
        "channel": "whatsapp",
        "protection_plan": {
            "plan_type": "proteccion_total",
            "protections_used": 1,
            "protections_allowed": 2,
            "status": "active",
            "valid_until": "2026-12-31",
            "reset_cycle_days": 365,
            "requires_manual_review": True,
        },
    }
    assert row["plan_type"] == "proteccion_total"
    assert row["protections_used"] == 1
    assert row["protections_allowed"] == 2


def test_aggregate_outcomes(tmp_path):
    log_path = tmp_path / "outcomes.csv"
    decision = _sample_decision()
    record_outcome(
        decision,
        "resolved",
        metadata={
            "protection_plan": {
                "plan_type": "proteccion_total",
                "protections_used": 1,
                "protections_allowed": 2,
                "status": "active",
                "valid_until": "2026-12-31",
                "reset_cycle_days": 365,
                "requires_manual_review": True,
            }
        },
        log_path=log_path,
    )
    record_outcome(
        _sample_decision(placa="TEST-002", action="payment_reminder", template="PIA_RECORDATORIO"),
        "escalated",
        log_path=log_path,
    )
    df = load_outcomes(log_path)
    features = aggregate_outcomes(df, windows=(30,))
    assert {
        "placa",
        "outcomes_total",
        "last_outcome_at",
        "investigate_consumption_resolved_all",
        "last_plan_type",
        "last_plan_status",
        "last_plan_valid_until",
        "last_plan_reset_cycle_days",
        "last_plan_requires_manual_review",
        "protections_remaining",
    }.issubset(
        set(features.columns)
    )
    row = features[features["placa"] == "TEST-001"].iloc[0]
    assert row["outcomes_total"] >= 1
    assert row["investigate_consumption_resolved_all"] == 1
    assert row["last_plan_type"] == "proteccion_total"
    assert row["last_plan_status"] == "active"
    assert row["last_plan_valid_until"] == "2026-12-31"
    assert row["last_plan_reset_cycle_days"] == 365
    assert bool(row["last_plan_requires_manual_review"]) is True
    assert row["protections_remaining"] == 1
