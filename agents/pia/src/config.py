"""Configuración central para el agente PIA."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass(frozen=True)
class RiskBandThresholds:
    low: float = 0.6
    medium: float = 0.75
    high: float = 0.85


@dataclass(frozen=True)
class TriggerConfig:
    coverage_low_14d: float = 0.65
    coverage_low_30d: float = 0.75
    downtime_high_hours_14d: float = 120.0
    cooldown_days_essential: int = 60
    cooldown_days_total: int = 30


@dataclass(frozen=True)
class AmountThresholds:
    overpayment_margin: float = -100.0  # arrears <= -100 MXN => prepago
    arrears_tolerance: float = 50.0     # tolerancia para considerar saldo al día


@dataclass(frozen=True)
class IntentAliases:
    balance_inquiry: Tuple[str, ...] = (
        "balance_inquiry",
        "consulta_saldo",
        "saldo",
        "balance",
    )
    advance_payment: Tuple[str, ...] = (
        "advance_payment",
        "pago_anticipado",
        "prepago",
        "prepayment",
    )
    payment_promise: Tuple[str, ...] = (
        "payment_promise",
        "promesa_pago",
        "promesa_de_pago",
        "promesa",
        "promise_to_pay",
    )
    document_support: Tuple[str, ...] = (
        "document_support",
        "soporte_documental",
        "documentos",
        "document_request",
        "solicitud_documentos",
    )


@dataclass(frozen=True)
class MessagingTemplate:
    name: str
    description: str
    placeholders: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class PIAAgentConfig:
    risk_thresholds: RiskBandThresholds = RiskBandThresholds()
    triggers: TriggerConfig = TriggerConfig()
    amounts: AmountThresholds = AmountThresholds()
    intents: IntentAliases = IntentAliases()
    whatsapp_templates: List[MessagingTemplate] = field(
        default_factory=lambda: [
            MessagingTemplate(
                name="PIA_OPCIONES",
                description="Recordatorio de pago con opciones de flexibilidad",
                placeholders=["nombre", "monto", "escenario1"],
            ),
            MessagingTemplate(
                name="PIA_RECORDATORIO",
                description="Recordatorio de pago regular",
                placeholders=["nombre", "referencia", "fecha"],
            ),
            MessagingTemplate(
                name="PIA_PROMESA",
                description="Confirmación de promesa de pago",
                placeholders=["nombre", "monto", "fecha_promesa"],
            ),
            MessagingTemplate(
                name="PIA_DOCUMENTOS",
                description="Entrega de documentación solicitada",
                placeholders=["nombre", "documento"],
            ),
            MessagingTemplate(
                name="PIA_CONSUMO",
                description="Investigación de consumo de GNV vs. telemetría",
                placeholders=["nombre", "placa", "alerta"],
            ),
            MessagingTemplate(
                name="PIA_FALLA",
                description="Escalamiento por fallas críticas detectadas",
                placeholders=["nombre", "placa", "diagnostico"],
            ),
        ]
    )


DEFAULT_CONFIG = PIAAgentConfig()
