"""Pydantic schemas for protection equilibrium service."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ProtectionEvaluateRequest(BaseModel):
    market: str = Field(..., example="edomex")
    balance: float = Field(..., gt=0)
    payment: float = Field(..., gt=0)
    term_months: int = Field(..., gt=0)

    restructures_used: int = Field(0, ge=0)
    restructures_allowed: Optional[int] = Field(None, ge=0)

    plan_type: Optional[str] = Field(
        None,
        description="Tipo de plan de protección contratado (opcional)",
        example="proteccion_total",
    )
    protections_used: Optional[int] = Field(
        None,
        ge=0,
        description="Número de protecciones ya utilizadas en el contrato",
    )
    protections_allowed: Optional[int] = Field(
        None,
        ge=0,
        description="Número máximo de protecciones contratadas para el plan",
    )

    has_consumption_gap: bool = False
    has_fault_alert: bool = False
    has_delinquency_flag: bool = False
    has_recent_promise_break: bool = False
    telematics_ok: bool = True

    log_outcome: bool = True
    decision_action: Optional[str] = Field(None, description="Original PIA decision action")
    decision_reason: Optional[str] = None
    decision_placa: Optional[str] = Field(None, description="Placa asociada para logging opcional")
    decision_risk_band: Optional[str] = Field(None, description="Banda de riesgo PIA para logging opcional")
    notes: str = ""
    metadata: Optional[Dict[str, Any]] = None
    requires_manual_review: Optional[bool] = Field(
        None,
        description="Si se activa, fuerza que los escenarios se marquen para revisión manual",
    )


class ProtectionScenario(BaseModel):
    type: str
    params: Dict[str, Any]
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


class ProtectionEvaluateResponse(BaseModel):
    context: Dict[str, Any]
    policy: Dict[str, Any]
    scenarios: List[ProtectionScenario]
    viable: List[ProtectionScenario]


class ProtectionEvaluateSummaryResponse(ProtectionEvaluateResponse):
    narrative: Optional[str] = Field(
        default=None,
        description="Texto narrativo generado por el LLM para explicar los escenarios de protección.",
    )
    narrative_context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Contexto enviado al LLM (para auditoría o debugging).",
    )
