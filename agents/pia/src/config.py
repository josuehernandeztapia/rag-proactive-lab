"""Configuración central para el agente PIA."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


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
class GNVConfig:
    min_overprice_per_liter: float = 5.0
    liters_fallback_days: int = 30
    epsilon_liters: float = 1e-6


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
class MemoryConfig:
    enabled: bool = True
    repeat_cooldown_hours: int = 24
    suppress_actions: Tuple[str, ...] = (
        "offer_protection",
        "payment_reminder",
        "investigate_consumption",
    )
    annotate_actions: Tuple[str, ...] = (
        "provide_balance",
        "provide_documents",
        "record_payment_promise",
    )
    escalate_action: str = "check_in"
    escalate_template: str = "PIA_RECORDATORIO"
    escalate_reason: str = "Seguimiento manual: respuesta reciente similar"


@dataclass(frozen=True)
class FollowUpConfig:
    protection_followup_hours: int = 72


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
    gnv: GNVConfig = GNVConfig()
    memory: MemoryConfig = MemoryConfig()
    followup: FollowUpConfig = FollowUpConfig()
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
            MessagingTemplate(
                name="PIA_SEGUIMIENTO",
                description="Seguimiento manual cuando la protección no se ejecuta",
                placeholders=["nombre", "motivo", "paso_siguiente"],
            ),
        ]
    )


_CONFIG_ENV_VAR = "PIA_CONFIG_PATH"


def _resolve_external_config_path() -> Optional[Path]:
    raw = os.getenv(_CONFIG_ENV_VAR)
    if not raw:
        return None
    try:
        return Path(raw).expanduser().resolve()
    except Exception:
        return None


def _load_external_overrides() -> Dict[str, Any]:
    path = _resolve_external_config_path()
    if not path or not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _merge_dataclass(instance: Any, updates: Dict[str, Any]) -> Any:
    if not is_dataclass(instance) or not isinstance(updates, dict):
        return instance
    kwargs: Dict[str, Any] = {}
    for field_info in fields(instance):
        name = field_info.name
        current_value = getattr(instance, name)
        if name not in updates:
            kwargs[name] = current_value
            continue
        new_value = updates[name]
        if is_dataclass(current_value):
            kwargs[name] = _merge_dataclass(current_value, new_value if isinstance(new_value, dict) else {})
        elif isinstance(current_value, list):
            if not isinstance(new_value, list):
                kwargs[name] = current_value
                continue
            if current_value and is_dataclass(current_value[0]):
                elem_cls = type(current_value[0])
                kwargs[name] = [elem_cls(**item) if isinstance(item, dict) else item for item in new_value]
            else:
                kwargs[name] = list(new_value)
        elif isinstance(current_value, tuple):
            if isinstance(new_value, (list, tuple)):
                kwargs[name] = tuple(new_value)
            else:
                kwargs[name] = current_value
        else:
            kwargs[name] = new_value
    return type(instance)(**kwargs)


def _build_config(overrides: Optional[Dict[str, Any]] = None) -> PIAAgentConfig:
    base = PIAAgentConfig()
    data = overrides if overrides is not None else _load_external_overrides()
    if not data:
        return base
    return _merge_dataclass(base, data)


_CURRENT_CONFIG = _build_config()
DEFAULT_CONFIG = _CURRENT_CONFIG


def get_config() -> PIAAgentConfig:
    return _CURRENT_CONFIG


def reload_config(overrides: Optional[Dict[str, Any]] = None) -> PIAAgentConfig:
    global _CURRENT_CONFIG, DEFAULT_CONFIG
    _CURRENT_CONFIG = _build_config(overrides)
    DEFAULT_CONFIG = _CURRENT_CONFIG
    return _CURRENT_CONFIG


def config_asdict(config: Optional[PIAAgentConfig] = None) -> Dict[str, Any]:
    return asdict(config or _CURRENT_CONFIG)


__all__ = [
    "RiskBandThresholds",
    "TriggerConfig",
    "AmountThresholds",
    "GNVConfig",
    "IntentAliases",
    "MemoryConfig",
    "FollowUpConfig",
    "MessagingTemplate",
    "PIAAgentConfig",
    "get_config",
    "reload_config",
    "config_asdict",
    "DEFAULT_CONFIG",
]
