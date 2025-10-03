from app.api import _build_protection_context
from app.schemas.protection import ProtectionEvaluateRequest


def test_build_protection_context_uses_dummy_contract():
    payload = ProtectionEvaluateRequest(
        market="edomex",
        balance=500_000,
        payment=18_000,
        term_months=48,
        metadata={"placa": "DEMO-001"},
    )

    context = _build_protection_context(payload)

    assert context.plan_type == "proteccion_total"
    assert context.protections_allowed == 3
    assert context.protections_used == 1
    assert context.requires_manual_review is False


def test_build_protection_context_respects_payload_overrides():
    payload = ProtectionEvaluateRequest(
        market="edomex",
        balance=500_000,
        payment=18_000,
        term_months=48,
        plan_type="custom_plan",
        protections_allowed=5,
        protections_used=2,
        metadata={"placa": "DEMO-001"},
    )

    context = _build_protection_context(payload)

    assert context.plan_type == "custom_plan"
    assert context.protections_allowed == 5
    assert context.protections_used == 2


def test_build_protection_context_marks_manual_review():
    payload = ProtectionEvaluateRequest(
        market="edomex",
        balance=480_000,
        payment=17_500,
        term_months=45,
        metadata={"placa": "DEMO-005"},
    )

    context = _build_protection_context(payload)

    assert context.plan_type == "proteccion_ilimitada"
    assert context.requires_manual_review is True

