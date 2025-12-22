#!/usr/bin/env python3
"""Smart Consolidation - Evita alertas duplicadas entre agentes sin duplicar lógica.

Coordinador ligero file-based que:
- Previene spam de alertas al mismo operador
- Consolida múltiples risks en una sola alerta enriquecida
- Mantiene hierarchy: HASE > PIA > Guardian por severidad
- No duplica lógica de negocio ni infrastructure
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Set

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SHARED_STATE_PATH = PROJECT_ROOT / 'data' / 'shared' / 'risk_consolidation_state.json'

# Agent hierarchy por severidad (orden de ownership)
AGENT_HIERARCHY = {
    'hase': 1,      # Default risk - máxima severidad
    'pia': 2,       # Portfolio risk - alta severidad
    'guardian': 3   # Operational risk - moderada severidad
}

# Cooldown periods por tipo de alerta (evita spam)
COOLDOWN_PERIODS = {
    'default_risk': timedelta(hours=6),     # Default risk: 6h cooldown
    'portfolio_risk': timedelta(hours=4),   # Portfolio risk: 4h cooldown
    'operational_risk': timedelta(hours=2), # Operational: 2h cooldown
    'consolidated': timedelta(hours=8),     # Consolidated: 8h cooldown
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert value to float."""
    if value is None:
        return default
    if isinstance(value, (int, float)) and not (isinstance(value, float) and math.isnan(value)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_shared_state() -> Dict[str, Any]:
    """Carga estado compartido de riesgo, crea archivo si no existe."""
    SHARED_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not SHARED_STATE_PATH.exists():
        return {}

    try:
        with SHARED_STATE_PATH.open('r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return {}


def save_shared_state(state: Dict[str, Any]) -> None:
    """Guarda estado compartido de riesgo."""
    SHARED_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SHARED_STATE_PATH.open('w', encoding='utf-8') as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def update_risk_context(placa: str, agent: str, risk_data: Dict[str, Any]) -> None:
    """Actualiza context de riesgo para una placa desde cualquier agente."""
    state = load_shared_state()

    if placa not in state:
        state[placa] = {
            'risk_context': {},
            'active_risks': {},
            'last_consolidated_alert': None,
            'last_agent_alerts': {}
        }

    placa_state = state[placa]

    # Actualizar context de riesgo
    placa_state['risk_context'].update({
        f'{agent}_{k}': v for k, v in risk_data.items()
    })
    placa_state['risk_context']['last_updated'] = datetime.now(timezone.utc).isoformat()

    save_shared_state(state)


def should_send_alert(
    placa: str,
    agent: str,
    risk_type: str,
    risk_score: float,
    risk_data: Optional[Dict[str, Any]] = None
) -> tuple[bool, Optional[str]]:
    """
    Determina si un agente debe enviar alerta o si debe consolidar.

    Returns:
        (should_send: bool, reason: Optional[str])
        - should_send=True: Agente puede enviar alerta
        - should_send=False: No enviar (ya enviada recientemente o otro agente owns)
        - reason: Explicación de la decisión
    """
    state = load_shared_state()
    now = datetime.now(timezone.utc)

    if placa not in state:
        # Primera vez para esta placa, siempre permitir
        return True, "first_alert_for_placa"

    placa_state = state[placa]

    # 1. Check cooldown para este tipo de riesgo
    cooldown_key = f'last_{risk_type}_alert'
    if cooldown_key in placa_state:
        try:
            last_alert = datetime.fromisoformat(placa_state[cooldown_key])
            cooldown_period = COOLDOWN_PERIODS.get(risk_type, timedelta(hours=2))

            if now - last_alert < cooldown_period:
                return False, f"cooldown_active_{risk_type}"
        except (ValueError, TypeError):
            pass

    # 2. Check si hay alertas activas de otros agentes
    active_agents = _get_active_agents(placa_state, now)

    if not active_agents:
        # No hay otros agentes activos, proceder
        return True, "no_other_active_agents"

    # 3. Determinar owner por hierarchy
    current_priority = AGENT_HIERARCHY.get(agent, 999)
    highest_priority_agent = min(active_agents, key=lambda a: AGENT_HIERARCHY.get(a, 999))
    highest_priority = AGENT_HIERARCHY.get(highest_priority_agent, 999)

    if current_priority <= highest_priority:
        # Este agente tiene igual o mayor prioridad
        if _should_consolidate(placa_state, agent, risk_score):
            return True, f"consolidating_with_{highest_priority_agent}"
        else:
            return True, "highest_priority_agent"
    else:
        # Otro agente tiene mayor prioridad
        return False, f"deferred_to_{highest_priority_agent}"


def _get_active_agents(placa_state: Dict[str, Any], now: datetime) -> Set[str]:
    """Obtiene agentes con alertas activas (dentro del período de consolidation)."""
    active_agents = set()

    last_alerts = placa_state.get('last_agent_alerts', {})
    for agent, last_alert_str in last_alerts.items():
        try:
            last_alert = datetime.fromisoformat(last_alert_str)
            # Considerar "activo" si alertó en las últimas 2 horas
            if now - last_alert < timedelta(hours=2):
                active_agents.add(agent)
        except (ValueError, TypeError):
            continue

    return active_agents


def _should_consolidate(placa_state: Dict[str, Any], agent: str, risk_score: float) -> bool:
    """Determina si se debe crear una alerta consolidada."""
    risk_context = placa_state.get('risk_context', {})

    # Buscar múltiples risks activos
    high_risks = []

    if 'hase_enhanced_default_risk' in risk_context:
        default_risk = _safe_float(risk_context['hase_enhanced_default_risk'])
        if default_risk > 0.7:
            high_risks.append('default_risk')

    if 'pia_overall_portfolio_risk' in risk_context:
        portfolio_risk = _safe_float(risk_context['pia_overall_portfolio_risk'])
        if portfolio_risk > 0.7:
            high_risks.append('portfolio_risk')

    if 'guardian_operational_stress' in risk_context:
        operational_risk = _safe_float(risk_context['guardian_operational_stress'])
        if operational_risk > 0.6:
            high_risks.append('operational_risk')

    # Consolidar si hay 2+ risks altos
    return len(high_risks) >= 2


def mark_alert_sent(
    placa: str,
    agent: str,
    risk_type: str,
    risk_score: float,
    was_consolidated: bool = False
) -> None:
    """Marca que un agente envió alerta para una placa."""
    state = load_shared_state()
    now = datetime.now(timezone.utc).isoformat()

    if placa not in state:
        state[placa] = {
            'risk_context': {},
            'active_risks': {},
            'last_consolidated_alert': None,
            'last_agent_alerts': {}
        }

    placa_state = state[placa]

    # Marcar alerta específica por tipo
    placa_state[f'last_{risk_type}_alert'] = now

    # Marcar alerta por agente
    placa_state['last_agent_alerts'][agent] = now

    # Si fue consolidada, marcar timestamp
    if was_consolidated:
        placa_state['last_consolidated_alert'] = now

    # Actualizar risk score actual
    placa_state['active_risks'][f'{agent}_{risk_type}'] = {
        'score': risk_score,
        'timestamp': now
    }

    save_shared_state(state)


def get_consolidation_context(placa: str) -> Optional[Dict[str, Any]]:
    """Obtiene context completo de riesgo para generar alerta consolidada."""
    state = load_shared_state()

    if placa not in state:
        return None

    placa_state = state[placa]
    risk_context = placa_state.get('risk_context', {})
    active_risks = placa_state.get('active_risks', {})

    # Construir context consolidado
    consolidation = {
        'placa': placa,
        'risks': {},
        'factors': [],
        'recommendations': [],
        'severity': 'MEDIUM'
    }

    # HASE context
    if 'hase_enhanced_default_risk' in risk_context:
        default_risk = _safe_float(risk_context['hase_enhanced_default_risk'])
        if default_risk > 0.7:
            consolidation['risks']['default_risk'] = {
                'score': default_risk,
                'agent': 'HASE',
                'description': f'Riesgo de default: {default_risk:.3f}'
            }
            if default_risk > 0.8:
                consolidation['severity'] = 'HIGH'

    # PIA context
    if 'pia_overall_portfolio_risk' in risk_context:
        portfolio_risk = _safe_float(risk_context['pia_overall_portfolio_risk'])
        if portfolio_risk > 0.7:
            consolidation['risks']['portfolio_risk'] = {
                'score': portfolio_risk,
                'agent': 'PIA',
                'description': f'Riesgo de cartera: {portfolio_risk:.3f}'
            }

    # Guardian context
    if 'guardian_operational_stress' in risk_context:
        operational_risk = _safe_float(risk_context['guardian_operational_stress'])
        if operational_risk > 0.6:
            consolidation['risks']['operational_risk'] = {
                'score': operational_risk,
                'agent': 'Guardian',
                'description': f'Stress operacional: {operational_risk:.3f}'
            }

    # Solo devolver si hay múltiples risks
    if len(consolidation['risks']) >= 2:
        return consolidation

    return None


def generate_consolidated_message(consolidation_context: Dict[str, Any]) -> str:
    """Genera mensaje consolidado inteligente."""
    placa = consolidation_context['placa']
    risks = consolidation_context['risks']
    severity = consolidation_context['severity']

    # Header basado en severidad
    if severity == 'HIGH':
        header = f"🚨 ALERTA CRÍTICA CONSOLIDADA - {placa}"
    else:
        header = f"⚠️ ALERTA MÚLTIPLE - {placa}"

    # Risk descriptions
    risk_lines = []
    for risk_type, risk_data in risks.items():
        agent = risk_data['agent']
        description = risk_data['description']
        risk_lines.append(f"• {agent}: {description}")

    # Determinar owner principal (highest severity)
    owner_agent = "COORDINADO"
    if 'default_risk' in risks:
        owner_agent = "HASE"
    elif 'portfolio_risk' in risks:
        owner_agent = "PIA"
    elif 'operational_risk' in risks:
        owner_agent = "Guardian"

    message = f"""{header}

🎯 MÚLTIPLES FACTORES DE RIESGO DETECTADOS:
{chr(10).join(risk_lines)}

💡 IMPACTO CONSOLIDADO:
Combinación de factores indica situación de alto riesgo que requiere atención inmediata y coordinada.

🎯 ACCIÓN RECOMENDADA:
Contacto prioritario con operador para revisión integral: situación financiera + patrones operacionales + comportamiento de manejo.

👑 Owner Principal: {owner_agent}
⏰ Siguiente revisión: 8 horas

---
Alerta consolidada automática para evitar duplicación
Timestamp: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"""

    return message


# Funciones de conveniencia para cada agente
def should_hase_send_alert(placa: str, default_risk: float, risk_data: Dict[str, Any] = None) -> tuple[bool, Optional[str]]:
    """Wrapper para HASE."""
    if risk_data:
        update_risk_context(placa, 'hase', risk_data)
    return should_send_alert(placa, 'hase', 'default_risk', default_risk, risk_data)


def should_pia_send_alert(placa: str, portfolio_risk: float, risk_data: Dict[str, Any] = None) -> tuple[bool, Optional[str]]:
    """Wrapper para PIA."""
    if risk_data:
        update_risk_context(placa, 'pia', risk_data)
    return should_send_alert(placa, 'pia', 'portfolio_risk', portfolio_risk, risk_data)


def should_guardian_send_alert(placa: str, operational_risk: float, risk_data: Dict[str, Any] = None) -> tuple[bool, Optional[str]]:
    """Wrapper para Guardian."""
    if risk_data:
        update_risk_context(placa, 'guardian', risk_data)
    return should_send_alert(placa, 'guardian', 'operational_risk', operational_risk, risk_data)


def mark_hase_alert_sent(placa: str, default_risk: float, consolidated: bool = False) -> None:
    """Wrapper para HASE."""
    mark_alert_sent(placa, 'hase', 'default_risk', default_risk, consolidated)


def mark_pia_alert_sent(placa: str, portfolio_risk: float, consolidated: bool = False) -> None:
    """Wrapper para PIA."""
    mark_alert_sent(placa, 'pia', 'portfolio_risk', portfolio_risk, consolidated)


def mark_guardian_alert_sent(placa: str, operational_risk: float, consolidated: bool = False) -> None:
    """Wrapper para Guardian."""
    mark_alert_sent(placa, 'guardian', 'operational_risk', operational_risk, consolidated)