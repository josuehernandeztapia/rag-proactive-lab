#!/usr/bin/env python3
"""Genera mensajes de Guardian de Flota a partir de guardian_insights.csv."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd
import yaml
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config" / "guardian.yml"
DEFAULT_OUTBOX = ROOT / "reports" / "guardian_outbox.jsonl"

SEVERITY_ORDER = {"high": 0, "medium": 1, "warning": 1, "low": 2, "info": 3}
SEVERITY_LABELS = {
    "high": "Alta",
    "medium": "Media",
    "warning": "Advertencia",
    "low": "Baja",
    "info": "Informativa",
}
ALERT_SUMMARY = {
    "downtime": "presenta inactividad prolongada",
    "inactivity": "registró una caída marcada de actividad",
    "consumption": "presenta una caída relevante en consumo de GNV",
    "driving": "tiene hábitos de conducción riesgosos detectados",
    "dtc": "tiene un diagnóstico activo en el vehículo",
    "telemetry_health": "presenta pérdida prolongada de telemetría",
    "geofence": "está fuera de la zona autorizada",
    "off_hours_usage": "operó fuera del horario establecido",
    "energy": "presenta caída en estado de carga",
}
BASE_RECOMMENDATIONS = {
    "downtime": "Contacta al operador o a logística para confirmar la ruta y liberar la unidad.",
    "inactivity": "Pregunta al operador si la unidad sigue operando y ajusta programación si aplica.",
    "consumption": "Verifica con el operador posibles cambios de ruta o revisa fugas/adeudos de combustible.",
    "driving": "Agenda coaching breve con el operador y refuerza las recomendaciones de manejo seguro.",
    "dtc": "Monitorea el código y agenda una revisión ligera; si reaparece o escala, coordina con soporte técnico.",
    "telemetry_health": "Verifica alimentación, fusibles y antena del GO; solicita apoyo si persiste.",
    "geofence": "Confirma con logística si la salida estuvo autorizada y coordina regreso si aplica.",
    "off_hours_usage": "Valida con operaciones si la ruta fuera de horario fue autorizada y analiza bloquear el uso.",
    "energy": "Agenda recarga y revisa posibles incidencias con el cargador o ciclos incompletos.",
}
DTC_RECOMMENDATIONS = {
    "DEVICE-UNPLUGGED": "Pide al operador reconectar el dispositivo GO para recuperar telemetría en vivo.",
    "DEVICE-POWER-LOSS": "Revisa alimentación y fusibles del GO; si es recurrente, escala a soporte técnico.",
    "DEVICE-FW-RESTART": "Solo monitorea; la actualización de firmware concluyó de forma correcta.",
    "P0420": "Agenda inspección del catalizador para evitar pérdida de eficiencia y potencia.",
    "P0138": "Revisa el sensor de O2 (Banco 1 Sensor 2) y programa diagnóstico si persiste la alerta.",
}
HANDOFF_CONFIG = {
    "telemetry_health": {
        "keyword": "SOPORTE",
        "closing": "¿Necesitas apoyo para reconectar el GO? Responde SOPORTE y te ayudamos de inmediato.",
        "handoff": "postventa_soporte",
    },
    "geofence": {
        "keyword": "SOPORTE",
        "closing": "Si requieres validación o seguimiento responde SOPORTE y lo escalamos internamente.",
        "handoff": "postventa_soporte",
    },
    "off_hours_usage": {
        "keyword": "SOPORTE",
        "closing": "¿Quieres revisar posible bloqueo o seguimiento? Responde SOPORTE y damos aviso.",
        "handoff": "postventa_soporte",
    },
    "energy": {
        "keyword": "SOPORTE",
        "closing": "Responde SOPORTE si necesitas asistencia con la recarga o diagnóstico del cargador.",
        "handoff": "postventa_soporte",
    },
    "dtc": {
        "keyword": "POSTVENTA",
        "closing": "Responde POSTVENTA si deseas programar revisión o diagnóstico.",
        "handoff": "postventa_tecnico",
    },
    "downtime": {
        "keyword": "SOPORTE",
        "closing": "¿Requieres asistencia en sitio? Responde SOPORTE y lo notificamos al equipo.",
        "handoff": "postventa_soporte",
    },
    "consumption": {
        "keyword": "PIA",
        "closing": "Responde PIA si prefieres revisar protecciones o ajustar escenarios financieros.",
        "handoff": "pia",
    },
    "driving": {
        "keyword": "POSTVENTA",
        "closing": "¿Quieres que lo registremos y revisemos riesgo de garantía? Responde POSTVENTA y damos seguimiento.",
        "handoff": "postventa_tecnico",
    },
}
DEFAULT_HANDOFF = {
    "keyword": "SOPORTE",
    "closing": "¿Necesitas ayuda adicional? Responde SOPORTE y coordinamos al equipo.",
    "handoff": "postventa_soporte",
}
MESSAGE_TEMPLATE = (
    "👋 Hola {contact_name}.\n\n"
    "Guardian de Flota detectó que {summary}.\n\n"
    "- ⏱ Registro: {timestamp_label}\n"
    "- 📍 Placa: {placa}\n"
    "- 🔍 Detalle: {detail_line}\n"
    "{severity_line}\n"
    "Sugerencia: {recommendation}\n\n"
    "{closing_line}"
)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crea mensajes proactivos de Guardian de Flota")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Ruta al YAML de configuración (guardian.yml)")
    parser.add_argument("--insights", type=Path, default=None, help="CSV generado por build_insights.py")
    parser.add_argument("--outbox", type=Path, default=None, help="JSONL donde se guardan las alertas formateadas")
    parser.add_argument("--limit", type=int, default=20, help="Número máximo de alertas a generar")
    parser.add_argument("--alert-type", action="append", dest="alert_types", help="Filtra por tipo de alerta (puede repetirse)")
    parser.add_argument("--min-severity", choices=["info", "low", "medium", "high"], default="info", help="Filtra por severidad mínima")
    parser.add_argument("--contact-default", default="equipo", help="Nombre de contacto por defecto si no hay mapeo")
    parser.add_argument("--contacts-csv", type=Path, help="CSV con columna placa/contacto para personalizar el saludo")
    parser.add_argument("--contact-column", default="contact", help="Nombre de la columna con el contacto en contacts-csv")
    parser.add_argument("--reference-ts", default=None, help="Timestamp ISO para calcular tiempos relativos (default=ahora)")
    parser.add_argument("--dry-run", action="store_true", help="Solo imprime en consola, no escribe en el outbox")
    parser.add_argument("--json", action="store_true", help="Imprime el payload final en JSON además del texto")
    return parser.parse_args(list(argv) if argv is not None else None)


def load_config(path: Path) -> Dict[str, Any]:
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def merge_handoff_config(config: Dict[str, Any]) -> None:
    alerts_cfg = config.get("alerts", {})
    mapping: Dict[str, str] = alerts_cfg.get("autohandoff", {}) or {}
    rules: Dict[str, Dict[str, Any]] = alerts_cfg.get("autohandoff_rules", {}) or {}

    for alert_type, route in mapping.items():
        base = HANDOFF_CONFIG.setdefault(alert_type, DEFAULT_HANDOFF.copy())
        base["handoff"] = route
        rule = rules.get(route, {})
        if "auto_escalate" in rule:
            base["auto_escalate"] = bool(rule.get("auto_escalate"))
        if "channels" in rule:
            channels = rule.get("channels") or []
            base["channels"] = channels if isinstance(channels, list) else [channels]


def resolve_path(root: Path, path_value: Optional[str]) -> Optional[Path]:
    if not path_value:
        return None
    candidate = Path(path_value)
    if not candidate.is_absolute():
        candidate = (root / candidate).resolve()
    return candidate


def load_contacts(path: Optional[Path], contact_column: str) -> Dict[str, str]:
    if not path:
        return {}
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"[guardian:notifier] Advertencia: contactos {path} no encontrados", file=sys.stderr)
        return {}
    column = contact_column
    if column not in df.columns:
        print(f"[guardian:notifier] Columna {column} no encontrada en {path}; se ignora mapeo", file=sys.stderr)
        return {}
    mapping: Dict[str, str] = {}
    for _, row in df.iterrows():
        placa = str(row.get("placa") or "").strip()
        contact = str(row.get(column) or "").strip()
        if placa and contact:
            mapping[placa] = contact
    return mapping


def parse_timestamp(raw: Any, tz: ZoneInfo) -> Optional[datetime]:
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return None
    if isinstance(raw, datetime):
        dt = raw
    else:
        value = str(raw).strip()
        if not value:
            return None
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(value)
        except ValueError:
            try:
                dt = pd.to_datetime(value, utc=True).to_pydatetime()
            except Exception:
                return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=tz)
    else:
        dt = dt.astimezone(tz)
    return dt


def humanize_timestamp(event_dt: Optional[datetime], now: datetime) -> str:
    if event_dt is None:
        return "sin fecha registrada"
    delta = now - event_dt
    suffix = "hace"
    if delta.total_seconds() < 0:
        delta = -delta
        suffix = "en"
    seconds = delta.total_seconds()
    if seconds < 90:
        return f"{suffix} {int(seconds)} segundos"
    minutes = seconds / 60
    if minutes < 90:
        return f"{suffix} {int(round(minutes))} min"
    hours = minutes / 60
    if hours < 48:
        return f"{suffix} {int(round(hours))} h"
    days = hours / 24
    if days < 14:
        return f"{suffix} {int(round(days))} días"
    return event_dt.strftime("%d %b %Y %H:%M %Z")


def severity_line(severity: str) -> str:
    label = SEVERITY_LABELS.get(severity, severity.title())
    if severity in {"info", ""}:
        return ""
    emoji = "⚠️" if severity in {"high", "medium", "warning"} else "ℹ️"
    return f"- {emoji} Severidad: {label}"


def pick_recommendation(alert_type: str, code: Optional[str]) -> str:
    if alert_type == "dtc" and code:
        if code in DTC_RECOMMENDATIONS:
            return DTC_RECOMMENDATIONS[code]
    base = BASE_RECOMMENDATIONS.get(alert_type)
    if base:
        return base
    return "Monitorea la unidad y, si la condición persiste, coordina con el equipo correspondiente."


def clean_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float):
        if math.isnan(value):
            return None
        return float(value)
    if isinstance(value, (int, str, bool)):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return str(value)


def make_message(row: Dict[str, Any], now: datetime, tz: ZoneInfo, contact: str) -> Dict[str, Any]:
    alert_type = row.get("alert_type", "unknown")
    placa = row.get("placa", "-")
    summary_base = ALERT_SUMMARY.get(alert_type, "se registró una señal relevante")
    summary = f"la placa {placa} {summary_base}"
    event_dt = parse_timestamp(row.get("triggered_at"), tz)
    timestamp_label = humanize_timestamp(event_dt, now)
    detail_line = row.get("details") or "Sin detalle"
    severity = row.get("severity", "info")
    code = row.get("code") or None
    recommendation = pick_recommendation(alert_type, code)
    handoff_cfg = HANDOFF_CONFIG.get(alert_type, DEFAULT_HANDOFF)
    closing_line = handoff_cfg.get("closing") or DEFAULT_HANDOFF["closing"]
    message = MESSAGE_TEMPLATE.format(
        contact_name=contact or "equipo",
        summary=summary,
        timestamp_label=timestamp_label,
        placa=placa,
        detail_line=detail_line,
        severity_line=severity_line(str(severity)),
        recommendation=recommendation,
        closing_line=closing_line,
    )
    return {
        "message": message,
        "event_ts": event_dt.isoformat() if event_dt else None,
        "summary": summary,
        "recommendation": recommendation,
        "handoff_hint": handoff_cfg.get("handoff", DEFAULT_HANDOFF["handoff"]),
        "handoff_keyword": handoff_cfg.get("keyword", DEFAULT_HANDOFF["keyword"]),
        "auto_escalate": handoff_cfg.get("auto_escalate", False),
        "auto_channels": handoff_cfg.get("channels", []),
    }


def filter_by_severity(df: pd.DataFrame, min_severity: str) -> pd.DataFrame:
    rank = SEVERITY_ORDER.get(min_severity, 3)
    df = df.copy()
    df["_severity_order"] = df["severity"].map(lambda s: SEVERITY_ORDER.get(str(s), 4))
    return df[df["_severity_order"] <= rank]


def load_insights(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de insights: {path}")
    return pd.read_csv(path)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    merge_handoff_config(config)
    timezone_name = config.get("timezone", "UTC")
    try:
        tz = ZoneInfo(timezone_name)
    except Exception:
        tz = ZoneInfo("UTC")
    paths_cfg = config.get("paths", {})
    insights_path = args.insights or paths_cfg.get("output")
    outbox_path = args.outbox or paths_cfg.get("outbox") or DEFAULT_OUTBOX
    insights_path = resolve_path(ROOT, insights_path)
    outbox_path = resolve_path(ROOT, outbox_path)
    if insights_path is None:
        print("[guardian:notifier] No se pudo resolver la ruta de insights", file=sys.stderr)
        return 1
    df = load_insights(insights_path)
    if df.empty:
        print(f"[guardian:notifier] Sin registros en {insights_path}")
        return 0

    if args.alert_types:
        df = df[df["alert_type"].isin(args.alert_types)]
    df = filter_by_severity(df, args.min_severity)
    if df.empty:
        print("[guardian:notifier] No hay alertas que cumplan los filtros actuales")
        return 0

    df["_severity_order"] = df["severity"].map(lambda s: SEVERITY_ORDER.get(str(s), 4))
    df = df.sort_values(["_severity_order", "triggered_at"], ascending=[True, False])

    contacts_map = load_contacts(resolve_path(ROOT, args.contacts_csv) if args.contacts_csv else None, args.contact_column)
    now = parse_timestamp(args.reference_ts, tz) if args.reference_ts else datetime.now(tz)
    if now is None:
        now = datetime.now(tz)

    limit = max(args.limit, 1)
    selected = df.head(limit)

    entries = []
    for _, row in selected.iterrows():
        raw = row.to_dict()
        placa = str(raw.get("placa") or "").strip()
        contact = contacts_map.get(placa, args.contact_default)
        msg_bundle = make_message(raw, now, tz, contact)
        insight_payload = {key: clean_value(value) for key, value in raw.items() if not key.startswith("_")}
        entry = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "placa": placa,
            "alert_type": raw.get("alert_type"),
            "severity": raw.get("severity"),
            "message": msg_bundle["message"],
            "summary": msg_bundle["summary"],
            "recommendation": msg_bundle["recommendation"],
            "event_ts": msg_bundle["event_ts"],
            "insight": insight_payload,
            "contact": contact,
            "source": "guardian-notifier",
            "handoff_hint": msg_bundle.get("handoff_hint"),
            "handoff_keyword": msg_bundle.get("handoff_keyword"),
        }
        entries.append(entry)

    for entry in entries:
        print("=" * 80)
        print(entry["message"])
        if args.json:
            print("-" * 80)
            print(json.dumps(entry, ensure_ascii=False, indent=2))

    if args.dry_run:
        return 0

    if outbox_path is None:
        print("[guardian:notifier] No se configuró ruta de outbox", file=sys.stderr)
        return 1
    outbox_path.parent.mkdir(parents=True, exist_ok=True)
    with outbox_path.open("a", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"[guardian:notifier] Se agregaron {len(entries)} alertas a {outbox_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
