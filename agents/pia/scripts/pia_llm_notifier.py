#!/usr/bin/env python3
"""Genera alerts proactivas a partir de los features agregados de PIA."""

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
EMAIL_FALLBACK_PATH = REPORTS_DIR / 'pia_llm_email_fallback.log'
PIA_OUTBOX_DEFAULT = REPORTS_DIR / 'pia_llm_outbox.jsonl'

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from app import storage as _storage  # type: ignore
except Exception:
    _storage = None

# Smart Consolidation import
try:
    from agents.shared.smart_consolidation import (
        should_pia_send_alert,
        mark_pia_alert_sent,
        get_consolidation_context,
        generate_consolidated_message
    )
    SMART_CONSOLIDATION_ENABLED = True
except ImportError:
    SMART_CONSOLIDATION_ENABLED = False

from agents.pia.src.llm_service import feature_enabled, get_llm_service  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genera alertas narrativas usando el servicio LLM")
    parser.add_argument("--features", type=Path, default=ROOT / "data" / "processed" / "pia" / "pia_features_enhanced.csv", help="Ruta al CSV de features agregados")
    parser.add_argument("--limit", type=int, default=5, help="Número máximo de alertas a generar")
    parser.add_argument("--min-protections", type=float, default=None, help="Filtra casos con protecciones restantes por debajo de este valor")
    parser.add_argument("--reference-ts", default=datetime.now(timezone.utc).isoformat(), help="Timestamp de referencia para las alertas")
    parser.add_argument("--email-to", default=os.getenv('PIA_ALERTS_EMAIL_TO'), help="Lista de correos (separados por coma) que recibirán la alerta")
    parser.add_argument("--email-subject", default=os.getenv('PIA_ALERTS_EMAIL_SUBJECT', '[PIA] Alerta proactiva {placa}'), help="Asunto del correo, permite placeholders como {placa}")
    parser.add_argument("--email-from", default=os.getenv('PIA_ALERTS_EMAIL_FROM'), help="Remitente del correo (fallback a SMTP_USERNAME)")
    parser.add_argument("--pia-outbox", type=Path, default=None, help="Archivo JSONL para encolar alertas dirigidas a PIA (ej. reports/pia_llm_outbox.jsonl)")
    parser.add_argument("--contact-column", default=os.getenv('PIA_ALERTS_CONTACT_COLUMN'), help="Columna del CSV que contiene el contacto del operador (si existe)")
    parser.add_argument("--skip-email", action='store_true', help="No enviar correo aunque se configure email_to")
    parser.add_argument("--skip-pia", action='store_true', help="No escribir en outbox aunque se configure pia_outbox")
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
    sender = sender or os.getenv('SMTP_USERNAME') or os.getenv('PIA_ALERTS_EMAIL_FROM', 'alerts@example.com')
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


def _enqueue_pia_outbox(outbox: Path, payload: Dict[str, Any], result: Dict[str, Any]) -> None:
    entry = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'placa': payload.get('placa'),
        'contact': payload.get('contact'),
        'flags': payload.get('flags'),
        'metric_snapshot': payload.get('metric_snapshot'),
        'content': result.get('content'),
        'context': result.get('context'),
    }
    outbox.parent.mkdir(parents=True, exist_ok=True)
    with outbox.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + '\n')
    if _storage is not None:
        try:
            _storage.log_event('pia_alert', entry)  # type: ignore[attr-defined]
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
    subject_template = args.email_subject or '[PIA] Alerta proactiva {placa}'
    try:
        subject = subject_template.format(**{**payload, **context})
    except Exception:
        subject = subject_template

    if recipients and not args.skip_email:
        _send_email_alert(subject, content, recipients, sender=args.email_from)

    outbox_path = Path(args.pia_outbox) if args.pia_outbox else PIA_OUTBOX_DEFAULT
    if not args.skip_pia:
        _enqueue_pia_outbox(outbox_path, payload, result)


def _boolish(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (float, int)):
        if isinstance(value, float) and math.isnan(value):
            return False
        return value > 0
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _safe_number(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float) and math.isnan(value):
        return "-"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    return f"{number:,.0f}"


def build_payload(row: pd.Series, reference_ts: str, contact_column: Optional[str]) -> Dict[str, Any]:
    flags = {
        "protecciones_negativas": _boolish(row.get("protections_flag_negative")),
        "plan_expirado": _boolish(row.get("protections_flag_expired")),
        "requiere_revision_manual": _boolish(row.get("protections_flag_manual")),
    }
    last_outcome_at = row.get("last_outcome_at")
    last_outcome_desc = "Último outcome no disponible"
    if isinstance(last_outcome_at, str) and last_outcome_at:
        last_outcome_desc = f"Último outcome el {last_outcome_at}"
    protections_remaining = row.get("protections_remaining")
    metric_snapshot = []
    if "outcomes_total" in row and not pd.isna(row.get("outcomes_total")):
        metric_snapshot.append(f"outcomes_totales={_safe_number(row.get('outcomes_total'))}")
    if "days_since_last_outcome" in row and not pd.isna(row.get("days_since_last_outcome")):
        metric_snapshot.append(f"días_sin_movimiento={_safe_number(row.get('days_since_last_outcome'))}")
    if "investigate_consumption_resolved_all" in row and not pd.isna(row.get("investigate_consumption_resolved_all")):
        metric_snapshot.append(
            f"investigaciones_resueltas={_safe_number(row.get('investigate_consumption_resolved_all'))}"
        )
    last_tags = str(row.get("last_behaviour_tags") or "").strip()
    if last_tags:
        metric_snapshot.append(f"tags={last_tags}")
    last_notes = str(row.get("last_behaviour_notes") or "").strip()
    if last_notes:
        metric_snapshot.append(f"notas={last_notes}")

    # ENHANCEMENT: Añadir features híbridos si están disponibles
    if "overall_portfolio_risk" in row and not pd.isna(row.get("overall_portfolio_risk")):
        metric_snapshot.append(f"riesgo_cartera_híbrido={_safe_number(row.get('overall_portfolio_risk'))}")
    if "safety_risk_component" in row and not pd.isna(row.get("safety_risk_component")):
        metric_snapshot.append(f"riesgo_seguridad={_safe_number(row.get('safety_risk_component'))}")
    if "operational_risk_component" in row and not pd.isna(row.get("operational_risk_component")):
        metric_snapshot.append(f"riesgo_operacional={_safe_number(row.get('operational_risk_component'))}")
    if "telemetry_enhancement_score" in row and not pd.isna(row.get("telemetry_enhancement_score")):
        metric_snapshot.append(f"score_telemetría={_safe_number(row.get('telemetry_enhancement_score'))}")

    # ENHANCEMENT: Eventos específicos de telemetría
    safety_events = []
    if "harsh_brake_events" in row and not pd.isna(row.get("harsh_brake_events")) and row.get("harsh_brake_events") > 0:
        safety_events.append(f"{_safe_number(row.get('harsh_brake_events'))} frenadas_bruscas")
    if "overspeed_events" in row and not pd.isna(row.get("overspeed_events")) and row.get("overspeed_events") > 0:
        safety_events.append(f"{_safe_number(row.get('overspeed_events'))} excesos_velocidad")
    if safety_events:
        metric_snapshot.append(f"eventos_seguridad={', '.join(safety_events)}")

    operational_events = []
    if "idling_events" in row and not pd.isna(row.get("idling_events")) and row.get("idling_events") > 0:
        operational_events.append(f"{_safe_number(row.get('idling_events'))} ralentí")
    if "after_hours_events" in row and not pd.isna(row.get("after_hours_events")) and row.get("after_hours_events") > 0:
        operational_events.append(f"{_safe_number(row.get('after_hours_events'))} fuera_horario")
    if operational_events:
        metric_snapshot.append(f"eventos_operacionales={', '.join(operational_events)}")

    contact_value = None
    if contact_column and contact_column in row:
        contact_value = row.get(contact_column)
    elif "contact" in row:
        contact_value = row.get("contact")
    elif "contact_address" in row:
        contact_value = row.get("contact_address")

    return {
        "reference_ts": reference_ts,
        "placa": row.get("placa", "SIN_PLACA"),
        "contact": contact_value,
        "market": row.get("plaza_limpia") or row.get("market") or "-",
        "plan_type": row.get("last_plan_type") or "-",
        "plan_status": row.get("last_plan_status") or "-",
        "plan_valid_until": row.get("last_plan_valid_until") or "-",
        "protections_remaining": protections_remaining,
        "flags": flags,
        "last_outcome_desc": last_outcome_desc,
        "metric_snapshot": "; ".join(metric_snapshot) if metric_snapshot else "Sin métricas adicionales",
        "impact_projection": _build_impact_projection(row, protections_remaining),
        "recommended_action": _build_recommended_action(row),
    }


def _build_impact_projection(row: pd.Series, protections_remaining: Any) -> str:
    """Construye projection de impacto con context híbrido."""
    base_projection = "Sin acción la TIR podría deteriorarse por falta de protecciones activas."

    # Añadir context de riesgo híbrido si está disponible
    risk_factors = []
    if "safety_risk_component" in row and not pd.isna(row.get("safety_risk_component")) and row.get("safety_risk_component") > 0.5:
        risk_factors.append("patrones de manejo riesgosos")
    if "operational_risk_component" in row and not pd.isna(row.get("operational_risk_component")) and row.get("operational_risk_component") > 0.5:
        risk_factors.append("stress operacional elevado")
    if "overall_portfolio_risk" in row and not pd.isna(row.get("overall_portfolio_risk")) and row.get("overall_portfolio_risk") > 0.7:
        risk_factors.append("riesgo de cartera híbrido alto")

    if risk_factors:
        base_projection += f" Factores adicionales detectados: {', '.join(risk_factors)}."

    return base_projection


def _build_recommended_action(row: pd.Series) -> str:
    """Construye recomendación con context específico."""
    base_action = "Contactar al operador, validar evidencias y activar la protección viable."

    # Añadir recomendaciones específicas basadas en telemetría
    specific_actions = []
    if "harsh_brake_events" in row and not pd.isna(row.get("harsh_brake_events")) and row.get("harsh_brake_events") > 10:
        specific_actions.append("revisar hábitos de frenado")
    if "overspeed_events" in row and not pd.isna(row.get("overspeed_events")) and row.get("overspeed_events") > 5:
        specific_actions.append("monitorear velocidad")
    if "idling_events" in row and not pd.isna(row.get("idling_events")) and row.get("idling_events") > 20:
        specific_actions.append("optimizar tiempos de ralentí")
    if "after_hours_events" in row and not pd.isna(row.get("after_hours_events")) and row.get("after_hours_events") > 5:
        specific_actions.append("verificar uso fuera de horario")

    if specific_actions:
        base_action += f" Considerar también: {', '.join(specific_actions)}."

    return base_action


def main() -> int:
    args = parse_args()
    recipients = _parse_recipients(args.email_to) if not args.skip_email else []
    if not feature_enabled("alerts"):
        print("PIA_LLM_ALERTS está deshabilitado (setea la variable de entorno para habilitarlo)", file=sys.stderr)
        return 0
    service = get_llm_service()
    if service is None:
        print("Servicio LLM no inicializado. Ajusta PIA_LLM_MODE (template u openai).", file=sys.stderr)
        return 1
    if not args.features.exists():
        print(f"No se encontró el archivo de features: {args.features}", file=sys.stderr)
        return 1
    df = pd.read_csv(args.features)
    if df.empty:
        print("El archivo de features está vacío; no hay alertas que generar.")
        return 0
    mask = (
        df.get("protections_flag_negative", False).fillna(False)
        | df.get("protections_flag_expired", False).fillna(False)
        | df.get("protections_flag_manual", False).fillna(False)
    )
    filtered = df[mask]
    if args.min_protections is not None:
        filtered = filtered[filtered.get("protections_remaining", 0) <= args.min_protections]
    if filtered.empty:
        print("Sin banderas críticas en los features filtrados.")
        return 0
    limit_df = filtered.head(args.limit if args.limit > 0 else len(filtered))
    for _, row in limit_df.iterrows():
        placa = str(row.get("placa", "SIN_PLACA"))
        portfolio_risk = _safe_number(row.get("overall_portfolio_risk", 0))

        # Convertir a float para Smart Consolidation
        try:
            portfolio_risk_float = float(portfolio_risk.replace(",", ".")) if isinstance(portfolio_risk, str) else float(portfolio_risk)
        except (ValueError, TypeError):
            portfolio_risk_float = 0.0

        # Smart Consolidation check
        should_send = True
        consolidation_reason = "smart_consolidation_disabled"
        is_consolidated = False

        if SMART_CONSOLIDATION_ENABLED:
            # Preparar risk data para context sharing
            risk_data = {
                'overall_portfolio_risk': portfolio_risk_float,
                'core_financial_risk': _safe_number(row.get("core_financial_risk", 0)),
                'telemetry_enhancement_score': _safe_number(row.get("telemetry_enhancement_score", 0)),
                'safety_risk_component': _safe_number(row.get("safety_risk_component", 0)),
                'operational_risk_component': _safe_number(row.get("operational_risk_component", 0)),
                'risk_category': row.get("risk_category", "unknown")
            }

            should_send, consolidation_reason = should_pia_send_alert(placa, portfolio_risk_float, risk_data)

            if not should_send:
                print(f"--- PIA alerta para {placa} diferida: {consolidation_reason} ---")
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
                            "primary_agent": "pia"
                        }
                    }
                    is_consolidated = True
                    print(f"--- ALERTA CONSOLIDADA para {placa} (owner: PIA) ---")
                else:
                    # Fallback a alerta normal de PIA
                    payload = build_payload(row, args.reference_ts, args.contact_column)
                    result = service.render_alert(payload)
                    if result:
                        service.persist_alert(payload, result)
            else:
                # Alerta normal de PIA
                payload = build_payload(row, args.reference_ts, args.contact_column)
                result = service.render_alert(payload)
                if result:
                    service.persist_alert(payload, result)
        else:
            # Sin Smart Consolidation, comportamiento original
            payload = build_payload(row, args.reference_ts, args.contact_column)
            result = service.render_alert(payload)
            if result:
                service.persist_alert(payload, result)

        if not result:
            continue

        # Marcar alerta como enviada en Smart Consolidation
        if SMART_CONSOLIDATION_ENABLED:
            mark_pia_alert_sent(placa, portfolio_risk_float, is_consolidated)

        if not is_consolidated:
            payload = build_payload(row, args.reference_ts, args.contact_column)

        _deliver_alert(args, payload if not is_consolidated else {}, result, recipients)
        print(f"--- Alerta para {placa} ---")
        print(result["content"])
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
