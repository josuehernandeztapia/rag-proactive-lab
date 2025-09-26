"""Utilidades mínimas para el agente PIA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List

from .rules import decide_action

def _mock_hase_api(placa: str) -> Dict[str, Any]:
    return {"placa": placa, "risk_score": 0.72}

def _mock_simulate(payload: Dict[str, Any]) -> Dict[str, Any]:
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
                func=_mock_hase_api,
            ),
            PIATool(
                name="simulate_pia",
                description="Simular decisión PIA",
                func=_mock_simulate,
            ),
            PIATool(
                name="trigger_make_webhook",
                description="Invocar escenario Make (placeholder)",
                func=lambda data: "Webhook enviado",
            ),
        ]

    def simulate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return _mock_simulate(payload)

def build_pia_agent() -> PIAgent:
    return PIAgent()
