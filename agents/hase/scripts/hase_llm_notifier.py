#!/usr/bin/env python3
"""Genera alertas proactivas de riesgo de default a partir de HASE enhanced predictions."""

from __future__ import annotations

import argparse
import json
import math
import os
import smtplib
import sys
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
REPORTS_DIR = ROOT / 'reports'
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
EMAIL_FALLBACK_PATH = REPORTS_DIR / 'hase_llm_email_fallback.log'
HASE_OUTBOX_DEFAULT = REPORTS_DIR / 'hase_llm_outbox.jsonl'

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from app import storage as _storage  # type: ignore
except Exception:
    _storage = None

# Smart Consolidation import
try:
    from agents.shared.smart_consolidation import (
        should_hase_send_alert,
        mark_hase_alert_sent,
        get_consolidation_context,
        generate_consolidated_message
    )
    SMART_CONSOLIDATION_ENABLED = True
except ImportError:
    SMART_CONSOLIDATION_ENABLED = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genera alertas de riesgo de default usando HASE enhanced")
    parser.add_argument("--features", type=Path, default=ROOT / "data" / "processed" / "hase" / "enhanced_default_predictions.csv", help="Ruta al CSV de HASE enhanced")
    parser.add_argument("--limit", type=int, default=5, help="Número máximo de alertas a generar")
    parser.add_argument("--min-default-risk", type=float, default=0.7, help="Filtra casos con riesgo de default superior a este valor")
    parser.add_argument("--reference-ts", default=datetime.now(timezone.utc).isoformat(), help="Timestamp de referencia para las alertas")
    parser.add_argument("--email-to", default=os.getenv('HASE_ALERTS_EMAIL_TO'), help="Lista de correos (separados por coma) que recibirán la alerta")
    parser.add_argument("--email-subject", default=os.getenv('HASE_ALERTS_EMAIL_SUBJECT', '[HASE] Alerta riesgo default {placa}'), help="Asunto del correo, permite placeholders como {placa}")
    parser.add_argument("--email-from", default=os.getenv('HASE_ALERTS_EMAIL_FROM'), help="Remitente del correo (fallback a SMTP_USERNAME)")
    parser.add_argument("--hase-outbox", type=Path, default=None, help="Archivo JSONL para encolar alertas dirigidas a HASE")
    parser.add_argument("--contact-column", default=os.getenv('HASE_ALERTS_CONTACT_COLUMN'), help="Columna del CSV que contiene el contacto del operador")
    parser.add_argument("--skip-email", action='store_true', help="No enviar correo aunque se configure email_to")
    parser.add_argument("--skip-hase", action='store_true', help="No escribir en outbox aunque se configure hase_outbox")
    return parser.parse_args()


def _parse_recipients(value: str | None) -> list[str]:
    if not value:
        return []
    return [addr.strip() for addr in value.split(',') if addr.strip()]


def _log_email_fallback(subject: str, body: str, recipients: list[str]) -> None:
    entry = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'subject': subject,
        'recipients': recipients,
        'body': body,
    }
    with EMAIL_FALLBACK_PATH.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + '\n')


def _send_email_alert(subject: str, body: str, recipients: list[str], sender: str | None = None) -> None:
    if not recipients:
        return
    sender = sender or os.getenv('SMTP_USERNAME') or os.getenv('HASE_ALERTS_EMAIL_FROM', 'hase@example.com')
    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = sender
    msg['To'] = ', '.join(recipients)
    msg.set_content(body)

    smtp_host = os.getenv('SMTP_HOST')
    smtp_port = int(os.getenv('SMTP_PORT', '587'))
    smtp_username = os.getenv('SMTP_USERNAME')
    smtp_password = os.getenv('SMTP_PASSWORD')
    use_tls = os.getenv('SMTP_USE_TLS', '1').strip().lower() in {'1', 'true', 'yes', 'y'}

    if not smtp_host:
        _log_email_fallback(subject, body, recipients)
        return
    try:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=20) as smtp:
            if use_tls:
                smtp.starttls()
            if smtp_username and smtp_password:
                smtp.login(smtp_username, smtp_password)
            smtp.send_message(msg)
    except Exception:
        _log_email_fallback(subject, body, recipients)


def _enqueue_hase_outbox(outbox: Path, payload: Dict[str, Any], result: Dict[str, Any]) -> None:
    entry = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'placa': payload.get('placa'),
        'contact': payload.get('contact'),
        'default_risk_flags': payload.get('default_risk_flags'),
        'risk_metrics': payload.get('risk_metrics'),
        'content': result.get('content'),
        'context': result.get('context'),
    }
    outbox.parent.mkdir(parents=True, exist_ok=True)
    with outbox.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + '\n')
    if _storage is not None:
        try:
            _storage.log_event('hase_alert', entry)  # type: ignore[attr-defined]
        except Exception:
            pass


def _deliver_alert(
    args: argparse.Namespace,
    payload: Dict[str, Any],
    result: Dict[str, Any],
    recipients: list[str],
) -> None:
    content = result.get('content', '') or ''
    if not content:
        return
    context = result.get('context') or {}
    subject_template = args.email_subject or '[HASE] Alerta riesgo default {placa}'
    try:
        subject = subject_template.format(**{**payload, **context})
    except Exception:
        subject = subject_template

    if recipients and not args.skip_email:
        _send_email_alert(subject, content, recipients, sender=args.email_from)

    outbox_path = Path(args.hase_outbox) if args.hase_outbox else HASE_OUTBOX_DEFAULT
    if not args.skip_hase:
        _enqueue_hase_outbox(outbox_path, payload, result)


def _safe_number(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float) and math.isnan(value):
        return "-"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    return f"{number:.3f}"


def _boolish(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (float, int)):
        if isinstance(value, float) and math.isnan(value):
            return False
        return value > 0
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def build_payload(row: pd.Series, reference_ts: str, contact_column: Optional[str]) -> Dict[str, Any]:
    """Construye payload para HASE default risk alert."""
    # Flags de riesgo de default
    default_risk_flags = {
        "high_default_risk": _boolish(row.get("enhanced_default_risk", 0) > 0.7),
        "behavioral_risk": _boolish(row.get("behavioral_default_risk", 0) > 0.6),
        "moderate_risk": row.get("risk_category") == "medium_risk",
        "high_risk": row.get("risk_category") == "high_risk",
    }

    # Métricas de riesgo
    risk_metrics = []
    if "enhanced_default_risk" in row and not pd.isna(row.get("enhanced_default_risk")):
        risk_metrics.append(f"riesgo_default_híbrido={_safe_number(row.get('enhanced_default_risk'))}")
    if "core_default_risk" in row and not pd.isna(row.get("core_default_risk")):
        risk_metrics.append(f"riesgo_core={_safe_number(row.get('core_default_risk'))}")
    if "behavioral_default_risk" in row and not pd.isna(row.get("behavioral_default_risk")):
        risk_metrics.append(f"riesgo_comportamental={_safe_number(row.get('behavioral_default_risk'))}")
    if "confidence" in row and not pd.isna(row.get("confidence")):
        risk_metrics.append(f"confidence={_safe_number(row.get('confidence'))}")

    # Context adicional
    risk_factors = []
    if "consumption_consistency" in row and not pd.isna(row.get("consumption_consistency")) and row.get("consumption_consistency") < 0.5:
        risk_factors.append("consumo inconsistente")
    if "coverage_stability" in row and not pd.isna(row.get("coverage_stability")) and row.get("coverage_stability") < 0.5:
        risk_factors.append("cobertura inestable")
    if "unauthorized_usage_risk" in row and not pd.isna(row.get("unauthorized_usage_risk")) and row.get("unauthorized_usage_risk") > 0.6:
        risk_factors.append("uso no autorizado")
    if "fraud_risk" in row and not pd.isna(row.get("fraud_risk")) and row.get("fraud_risk") > 0.5:
        risk_factors.append("indicadores de fraude")

    contact_value = None
    if contact_column and contact_column in row:
        contact_value = row.get(contact_column)
    elif "contact" in row:
        contact_value = row.get("contact")

    return {
        "reference_ts": reference_ts,
        "placa": row.get("placa", "SIN_PLACA"),
        "contact": contact_value,
        "risk_category": row.get("risk_category") or "unknown",
        "enhanced_default_risk": row.get("enhanced_default_risk", 0),
        "default_risk_flags": default_risk_flags,
        "risk_metrics": "; ".join(risk_metrics) if risk_metrics else "Sin métricas disponibles",
        "risk_factors": "; ".join(risk_factors) if risk_factors else "Sin factores específicos detectados",
        "label_reason": row.get("label_reason", "Análisis automático HASE"),
        "impact_projection": _build_impact_projection(row),
        "recommended_action": _build_recommended_action(row),
    }


def _build_impact_projection(row: pd.Series) -> str:
    """Construye projection de impacto para default risk."""
    base_projection = "Sin intervención, existe alta probabilidad de incumplimiento en pagos."

    # Añadir context específico
    risk_factors = []
    if row.get("behavioral_default_risk", 0) > 0.6:
        risk_factors.append("patrones de comportamiento riesgosos")
    if row.get("fraud_risk", 0) > 0.5:
        risk_factors.append("indicadores de fraude")
    if row.get("operational_stress_risk", 0) > 0.7:
        risk_factors.append("stress operacional elevado")

    if risk_factors:
        base_projection += f" Factores identificados: {', '.join(risk_factors)}."

    return base_projection


def _build_recommended_action(row: pd.Series) -> str:
    """Construye recomendación específica para default risk."""
    base_action = "Contactar inmediatamente al operador para revisar situación financiera y operacional."

    # Añadir acciones específicas
    specific_actions = []
    if row.get("coverage_stability", 1) < 0.5:
        specific_actions.append("revisar estabilidad de cobertura")
    if row.get("consumption_consistency", 1) < 0.5:
        specific_actions.append("analizar patrones de consumo")
    if row.get("unauthorized_usage_risk", 0) > 0.6:
        specific_actions.append("investigar uso no autorizado")
    if row.get("fraud_risk", 0) > 0.5:
        specific_actions.append("escalar a investigación de fraude")

    if specific_actions:
        base_action += f" Acciones prioritarias: {', '.join(specific_actions)}."

    return base_action


def generate_simple_alert(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Genera alerta simple para HASE (sin LLM por ahora)."""
    placa = payload.get("placa", "SIN_PLACA")
    risk_level = payload.get("risk_category", "unknown")
    risk_score = payload.get("enhanced_default_risk", 0)

    content = f"""🚨 ALERTA HASE - RIESGO DE DEFAULT

Placa: {placa}
Nivel de Riesgo: {risk_level.upper()}
Score Híbrido: {risk_score:.3f}

📊 Métricas:
{payload.get('risk_metrics', 'Sin métricas')}

⚠️ Factores de Riesgo:
{payload.get('risk_factors', 'Sin factores específicos')}

💡 Razón del Alert:
{payload.get('label_reason', 'Análisis automático')}

📈 Proyección de Impacto:
{payload.get('impact_projection', 'Sin proyección')}

🎯 Acción Recomendada:
{payload.get('recommended_action', 'Contactar al operador')}

---
Generado por HASE Enhanced Default Prediction
Timestamp: {payload.get('reference_ts', datetime.now(timezone.utc).isoformat())}
"""

    return {
        "content": content,
        "context": {
            "alert_type": "default_risk",
            "risk_level": risk_level,
            "risk_score": risk_score,
        }
    }


def main() -> int:
    args = parse_args()
    recipients = _parse_recipients(args.email_to) if not args.skip_email else []

    if not args.features.exists():
        print(f"No se encontró el archivo de features: {args.features}", file=sys.stderr)
        return 1

    df = pd.read_csv(args.features)
    if df.empty:
        print("El archivo de features está vacío; no hay alertas que generar.")
        return 0

    # Filtrar por riesgo de default alto
    filtered = df[df.get("enhanced_default_risk", 0) >= args.min_default_risk]

    if filtered.empty:
        print(f"Sin registros con riesgo de default >= {args.min_default_risk}")
        return 0

    # Ordenar por riesgo descendente
    filtered = filtered.sort_values("enhanced_default_risk", ascending=False)
    limit_df = filtered.head(args.limit if args.limit > 0 else len(filtered))

    for _, row in limit_df.iterrows():
        placa = str(row.get("placa", "SIN_PLACA"))
        default_risk = _safe_float(row.get("enhanced_default_risk", 0))

        # Smart Consolidation check
        should_send = True
        consolidation_reason = "smart_consolidation_disabled"
        is_consolidated = False

        if SMART_CONSOLIDATION_ENABLED:
            # Preparar risk data para context sharing
            risk_data = {
                'enhanced_default_risk': default_risk,
                'core_default_risk': _safe_float(row.get("core_default_risk", 0)),
                'behavioral_default_risk': _safe_float(row.get("behavioral_default_risk", 0)),
                'risk_category': row.get("risk_category", "unknown"),
                'confidence': _safe_float(row.get("confidence", 0)),
                'fraud_risk': _safe_float(row.get("fraud_risk", 0)),
                'consumption_consistency': _safe_float(row.get("consumption_consistency", 1))
            }

            should_send, consolidation_reason = should_hase_send_alert(placa, default_risk, risk_data)

            if not should_send:
                print(f"--- HASE alerta para {placa} diferida: {consolidation_reason} ---")
                continue

            # Check si debe consolidar con otros agentes
            if "consolidating" in consolidation_reason:
                consolidation_context = get_consolidation_context(placa)
                if consolidation_context:
                    # Generar alerta consolidada
                    consolidated_message = generate_consolidated_message(consolidation_context)
                    result = {
                        "content": consolidated_message,
                        "context": {
                            "alert_type": "consolidated_risk",
                            "consolidation_context": consolidation_context,
                            "primary_agent": "hase"
                        }
                    }
                    is_consolidated = True
                    print(f"--- ALERTA CONSOLIDADA para {placa} (owner: HASE) ---")
                else:
                    # Fallback a alerta normal
                    payload = build_payload(row, args.reference_ts, args.contact_column)
                    result = generate_simple_alert(payload)
            else:
                # Alerta normal de HASE
                payload = build_payload(row, args.reference_ts, args.contact_column)
                result = generate_simple_alert(payload)
        else:
            # Sin Smart Consolidation, comportamiento original
            payload = build_payload(row, args.reference_ts, args.contact_column)
            result = generate_simple_alert(payload)

        if not result:
            continue

        # Marcar alerta como enviada en Smart Consolidation
        if SMART_CONSOLIDATION_ENABLED:
            mark_hase_alert_sent(placa, default_risk, is_consolidated)

        if not is_consolidated:
            payload = build_payload(row, args.reference_ts, args.contact_column)

        _deliver_alert(args, payload if not is_consolidated else {}, result, recipients)
        print(f"--- Alerta HASE para {placa} ---")
        print(result["content"])
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())