"""Utilidades mínimas para el agente PIA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List

from agents.hase.src import score_driver, score_payload

from .rules import decide_action


def _hase_consult(entry: Any = None, /, **extra: Any) -> Dict[str, Any]:
    """Consultar el stub HASE.

    Acepta ya sea un diccionario completo con métricas del operador o
    únicamente la placa como primer argumento posicional. Cualquier keyword
    extra se mezcla en el payload antes de invocar el servicio.
    """

    if isinstance(entry, dict):
        payload: Dict[str, Any] = entry.copy()
    else:
        payload = dict(extra)
        if entry is not None:
            payload.setdefault("placa", entry)
    if "placa" not in payload or not payload["placa"]:
        raise ValueError("placa requerida para consultar HASE")
    return score_payload(payload).to_dict()


def _simulate_with_rules(payload: Dict[str, Any]) -> Dict[str, Any]:
    return decide_action(payload).__dict__


@dataclass
class PIATool:
    name: str
    description: str
    func: Callable[..., Any]


class PIAgent:
    def __init__(self) -> None:
        self.tools: List[PIATool] = [
            PIATool(
                name="consult_hase",
                description="Consultar score HASE",
                func=_hase_consult,
            ),
            PIATool(
                name="simulate_pia",
                description="Simular decisión PIA",
                func=_simulate_with_rules,
            ),
            PIATool(
                name="trigger_make_webhook",
                description="Invocar escenario Make (placeholder)",
                func=lambda data: "Webhook enviado",
            ),
        ]

    def simulate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return _simulate_with_rules(payload)

    def consult_hase(self, placa: str, *, payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
        base = payload.copy() if payload else {}
        return score_driver(placa, payload=base).to_dict()


def build_pia_agent() -> PIAgent:
    return PIAgent()
