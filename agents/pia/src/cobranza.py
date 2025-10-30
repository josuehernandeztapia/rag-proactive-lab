"""Cobranza utilities for PIA.

Provide dataclasses and helpers to determine collection strategies, generate
personalised messages and escalate cases to human advisors when needed.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

try:
    from agents.whatsapp_outbox import append_message as enqueue_whatsapp_message
except Exception:  # pragma: no cover - fallback when not available
    enqueue_whatsapp_message = None  # type: ignore

DEFAULT_ALERTS_PATH = Path("reports/pia_cobranza_alerts.jsonl")


@dataclass
class CobranzaCase:
    placa: str
    dias_mora: int = 0
    monto_vencido: float = 0.0
    saldo_pendiente: float = 0.0
    intentos_contacto: int = 0
    ultima_respuesta: Optional[datetime] = None
    protection_offered: bool = False
    escalated_to_advisor: bool = False
    client_context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CobranzaStrategy:
    tone: str
    canal: str
    mensaje_template: str
    include_protection_offer: bool = False
    escalate_after_attempts: int = 0
    prepare_advisor_context: bool = False


COBRANZA_TEMPLATES: Dict[str, Any] = {
    "gentle_reminder": [
        "Hola {nombre}! 👋 Noté que tu pago de ${monto:.2f} está pendiente. ¿Todo bien? Si necesitas algo aquí estoy para ayudarte 😊",
        "¡Hey {nombre}! 🚛 Solo un recordatorio amigable sobre tu pago de ${monto:.2f}. Sabemos que a veces se complica, ¿te apoyo con algo?",
    ],
    "solution_oriented": [
        "Hola {nombre}, llevas {dias} días con un saldo pendiente de ${monto:.2f}. ¿Activamos tu protección para reestructurar? Podemos ajustar tus pagos 💪",
        "¡{nombre}! El pago de ${monto:.2f} tiene {dias} días de atraso. Tengo opciones para ayudarte, ¿hablamos de una reestructura? 🤝",
    ],
    "human_intervention": [
        "Case {placa}: {dias} días en mora y ${monto:.2f} vencidos. Intentos: {intentos}. Última respuesta: {ultima_respuesta}. Protección ofrecida: {protection_status}.",
    ],
    "protection_offer": "🛡️ ¿Sabías que puedes activar tu Protección para pausar pagos por {meses} meses? Te ayudo a configurarla en 2 minutos.",
}


def _alerts_path() -> Path:
    custom = os.getenv("PIA_COBRANZA_ALERTS_PATH")
    if custom:
        return Path(custom)
    return DEFAULT_ALERTS_PATH


def determinar_estrategia_cobranza(case: CobranzaCase) -> CobranzaStrategy:
    if case.dias_mora <= 3:
        return CobranzaStrategy(
            tone="suave",
            canal="whatsapp",
            mensaje_template="gentle_reminder",
            include_protection_offer=False,
            escalate_after_attempts=2,
        )
    if case.dias_mora <= 7:
        return CobranzaStrategy(
            tone="firme_colaborativo",
            canal="whatsapp_call",
            mensaje_template="solution_oriented",
            include_protection_offer=True,
            escalate_after_attempts=3,
        )
    return CobranzaStrategy(
        tone="escalation",
        canal="advisor_alert",
        mensaje_template="human_intervention",
        include_protection_offer=False,
        escalate_after_attempts=0,
        prepare_advisor_context=True,
    )


def _cliente_nombre(case: CobranzaCase) -> str:
    nombre = case.client_context.get("nombre") if case.client_context else None
    if not nombre:
        return case.client_context.get("alias", case.placa)
    return str(nombre)


def generar_mensaje_personalizado(template_key: str, case: CobranzaCase, *, include_protection: bool = False) -> str:
    plantilla = COBRANZA_TEMPLATES.get(template_key)
    nombre = _cliente_nombre(case)
    if isinstance(plantilla, list) and plantilla:
        base = random.choice(plantilla)
    elif isinstance(plantilla, str):
        base = plantilla
    else:
        base = "Hola {nombre}, tienes un saldo pendiente."  # fallback
    ultima_respuesta = (
        case.ultima_respuesta.astimezone(timezone.utc).isoformat()
        if isinstance(case.ultima_respuesta, datetime)
        else "sin respuesta"
    )
    mensaje = base.format(
        nombre=nombre,
        monto=case.monto_vencido,
        dias=case.dias_mora,
        placa=case.placa,
        intentos=case.intentos_contacto,
        ultima_respuesta=ultima_respuesta,
        protection_status="sí" if case.protection_offered else "no",
    )
    if include_protection:
        mensaje += "\n\n" + COBRANZA_TEMPLATES["protection_offer"].format(meses=2)
    return mensaje.strip()


def registrar_alerta_asesor(case: CobranzaCase, strategy: CobranzaStrategy) -> None:
    if case.escalated_to_advisor:
        return
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "placa": case.placa,
        "nombre": _cliente_nombre(case),
        "dias_mora": case.dias_mora,
        "monto_vencido": case.monto_vencido,
        "saldo_pendiente": case.saldo_pendiente,
        "intentos": case.intentos_contacto,
        "ultima_respuesta": case.ultima_respuesta.isoformat() if isinstance(case.ultima_respuesta, datetime) else None,
        "protection_offered": case.protection_offered,
        "strategy": asdict(strategy),
    }
    path = _alerts_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    case.escalated_to_advisor = True


def _quick_replies_for_strategy(strategy: CobranzaStrategy) -> Iterable[str]:
    if strategy.prepare_advisor_context:
        return ["Transferir a asesor"]
    if strategy.include_protection_offer:
        return ["Activar protección", "Necesito apoyo"]
    return []


def preparar_cobranza_payload(case: CobranzaCase) -> Dict[str, Any]:
    strategy = determinar_estrategia_cobranza(case)
    message = generar_mensaje_personalizado(
        strategy.mensaje_template,
        case,
        include_protection=strategy.include_protection_offer,
    )
    if enqueue_whatsapp_message:
        try:
            enqueue_whatsapp_message(
                agent="pia",
                message=message,
                contact=str(case.client_context.get("telefono")) if case.client_context and case.client_context.get("telefono") else None,
                placa=case.placa,
                quick_replies=_quick_replies_for_strategy(strategy),
                metadata={
                    "dias_mora": case.dias_mora,
                    "monto_vencido": case.monto_vencido,
                    "strategy": asdict(strategy),
                    "source": "pia_cobranza",
                },
            )
        except Exception:
            pass
    result = {
        "case": {
            "placa": case.placa,
            "dias_mora": case.dias_mora,
            "monto_vencido": case.monto_vencido,
            "saldo_pendiente": case.saldo_pendiente,
            "intentos_contacto": case.intentos_contacto,
            "ultima_respuesta": case.ultima_respuesta.isoformat() if isinstance(case.ultima_respuesta, datetime) else None,
            "protection_offered": case.protection_offered,
            "escalated_to_advisor": case.escalated_to_advisor,
            "client_context": case.client_context,
        },
        "strategy": asdict(strategy),
        "message": message,
    }
    if strategy.prepare_advisor_context and not case.escalated_to_advisor:
        registrar_alerta_asesor(case, strategy)
        result["advisor_alert"] = True
    else:
        result["advisor_alert"] = False
    return result
