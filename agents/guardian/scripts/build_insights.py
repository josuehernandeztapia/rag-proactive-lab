#!/usr/bin/env python3
"""Construye `data/guardian_insights.csv` combinando señales de telemetría.

Usage:
    python agents/guardian/scripts/build_insights.py
    python agents/guardian/scripts/build_insights.py --config config/guardian.yml

Lee staging de Geotab y el dataset PIA para emitir alertas accionables por placa."""

from __future__ import annotations

import argparse
import ast
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.metadata import write_metadata
DEFAULT_CONFIG = ROOT / "config" / "guardian.yml"
DTC_CATALOG_PATH = ROOT / "guardian" / "dtc_catalog.json"
GLOBAL_DTC_PATH = ROOT / "data" / "dtc_catalog.json"

Severity = str

def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)



def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genera métricas accionables para Guardian de Flota")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Ruta al archivo YAML de configuración")
    parser.add_argument("--output", type=Path, default=None, help="Ruta de salida (override del YAML)")
    return parser.parse_args(list(argv) if argv is not None else None)


def load_yaml(path: Path) -> Dict[str, Any]:
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_json(path: Path) -> Dict[str, Any]:
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_device_id(value: Any) -> str | None:
    if isinstance(value, str) and value.startswith("{'id':"):
        try:
            return ast.literal_eval(value)["id"]
        except Exception:
            return None
    return value if isinstance(value, str) else None


def extract_diagnostic_id(value: Any) -> str | None:
    if isinstance(value, str) and value.startswith("{'id':"):
        try:
            return ast.literal_eval(value)["id"]
        except Exception:
            return None
    return value if isinstance(value, str) else None


def load_device_map(devices_path: Path) -> dict[str, str]:
    try:
        df = pd.read_csv(devices_path)
    except FileNotFoundError:
        return {}
    if df.empty:
        return {}
    mapping: dict[str, str] = {}
    if {"id", "licensePlate"}.issubset(df.columns):
        id_col, plate_col = "id", "licensePlate"
    elif {"device_id", "license_plate"}.issubset(df.columns):
        id_col, plate_col = "device_id", "license_plate"
    else:
        id_col = next((c for c in df.columns if c.lower() in {"id", "device_id"}), None)
        plate_col = next((c for c in df.columns if c.lower() in {"license_plate", "licenseplate", "name"}), None)
    if id_col and plate_col:
        for _, row in df.iterrows():
            device_id = row.get(id_col)
            plate = row.get(plate_col)
            if isinstance(device_id, str) and isinstance(plate, str) and plate:
                mapping[device_id] = plate.strip()
    return mapping


def load_dtc_catalog() -> dict[str, Dict[str, Any]]:
    catalog: dict[str, Dict[str, Any]] = {}
    if DTC_CATALOG_PATH.exists():
        data = load_json(DTC_CATALOG_PATH)
        catalog = {k: v for k, v in data.get("diagnostics", {}).items() if isinstance(v, dict)}
    # Permite enriquecer descripciones a partir del catálogo global si existe un code
    global_catalog_codes: dict[str, Dict[str, Any]] = {}
    if GLOBAL_DTC_PATH.exists():
        data = load_json(GLOBAL_DTC_PATH)
        global_catalog_codes = {
            str(code).upper(): info
            for code, info in data.get("codes", {}).items()
            if isinstance(info, dict)
        }
    for diag_id, entry in catalog.items():
        code = entry.get("code")
        if code and isinstance(code, str):
            info = global_catalog_codes.get(code.upper())
            if info:
                entry.setdefault("description", info.get("description"))
                entry.setdefault("category", info.get("category"))
                entry.setdefault("severity", info.get("severity"))
    return catalog


def load_faults(faults_path: Path, device_map: dict[str, str]) -> pd.DataFrame:
    try:
        df = pd.read_csv(faults_path)
    except FileNotFoundError:
        return pd.DataFrame()
    if df.empty:
        return df
    df = df.copy()
    if "device" in df.columns:
        df["device_id"] = df["device"].apply(normalize_device_id)
    elif "device_id" not in df.columns:
        df["device_id"] = None

    if "diagnostic" in df.columns:
        df["diagnostic_id"] = df["diagnostic"].apply(extract_diagnostic_id)
    elif "diagnostic_id" not in df.columns:
        df["diagnostic_id"] = None

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    else:
        df["timestamp"] = pd.to_datetime(df.get("dateTime"), errors="coerce")

    if "placa" not in df.columns:
        df["placa"] = df["device_id"].map(device_map)

    df = df.dropna(subset=[col for col in ("placa", "diagnostic_id", "timestamp") if col in df.columns])
    return df


def _resolve_dtc_severity(row: pd.Series, config: dict[str, Any], entry: dict[str, Any]) -> Severity:
    cfg = config.get("alerts", {}).get("dtc", {})
    severity_map = cfg.get("severity_map", {}) if isinstance(cfg, dict) else {}
    fallback = cfg.get("fallback_severity", "medium") if isinstance(cfg, dict) else "medium"
    candidate: str | None = None
    diag_sev = row.get("diagnosticSeverity") or row.get("diagnostic_severity")
    if isinstance(entry, dict):
        entry_sev = entry.get("severity")
        if isinstance(entry_sev, str) and entry_sev:
            candidate = entry_sev
    if isinstance(diag_sev, str) and diag_sev:
        candidate = severity_map.get(diag_sev, candidate or severity_map.get("default"))
    general_sev = row.get("severity")
    if isinstance(general_sev, str) and general_sev:
        candidate = severity_map.get(general_sev, candidate)
    return candidate or fallback


def build_dtc_alerts(faults: pd.DataFrame, config: dict[str, Any], dtc_catalog: dict[str, Dict[str, Any]]) -> pd.DataFrame:
    if faults.empty:
        return pd.DataFrame(columns=["placa", "alert_type", "severity", "details", "triggered_at"])
    cfg = config.get("alerts", {}).get("dtc", {})
    include_pending = bool(cfg.get("include_pending", True))
    max_age_days = cfg.get("max_age_days")
    filtered = faults.copy()
    if not include_pending:
        filtered = filtered[filtered["faultState"].eq("Active")]
    max_ts = filtered["timestamp"].max()
    if pd.notna(max_ts) and isinstance(max_age_days, (int, float)) and max_age_days > 0:
        cutoff = max_ts - timedelta(days=float(max_age_days))
        filtered = filtered[filtered["timestamp"] >= cutoff]
    if filtered.empty:
        return pd.DataFrame(columns=["placa", "alert_type", "severity", "details", "triggered_at"])
    # Conserva el evento más reciente por placa + diagnóstico para evitar duplicados
    filtered = filtered.sort_values("timestamp", ascending=False)
    subset = filtered.groupby(["placa", "diagnostic_id"], as_index=False).first()
    records = []
    for _, row in subset.iterrows():
        diag_id = row["diagnostic_id"]
        entry = dtc_catalog.get(diag_id, {})
        code = entry.get("code") if isinstance(entry, dict) else None
        description = entry.get("description") if isinstance(entry, dict) else None
        recommendation = row.get("recommendation")
        details_parts = []
        if isinstance(code, str) and code:
            details_parts.append(f"Código {code}")
        if description:
            details_parts.append(description)
        fault_state = row.get("faultState") or row.get("fault_state")
        if isinstance(fault_state, str) and fault_state:
            details_parts.append(f"Estado: {fault_state}")
        if isinstance(recommendation, str) and recommendation:
            details_parts.append(recommendation)
        if not details_parts:
            details_parts.append(f"Diagnóstico {diag_id}")
        severity = _resolve_dtc_severity(row, config, entry)
        records.append(
            {
                "placa": row["placa"],
                "alert_type": "dtc",
                "severity": severity,
                "details": " | ".join(details_parts),
                "triggered_at": row["timestamp"].isoformat(),
                "code": code,
                "diagnostic_id": diag_id,
            }
        )
    return pd.DataFrame.from_records(records)


def build_driving_alerts(faults: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    alerts_cfg = config.get("alerts", {}).get("driving", {})
    event_ids = alerts_cfg.get("harsh_event_ids", []) if isinstance(alerts_cfg, dict) else []
    if not event_ids or faults.empty:
        return pd.DataFrame(columns=["placa", "alert_type", "severity", "details", "triggered_at"])
    lookback_days = alerts_cfg.get("lookback_days", 7)
    max_ts = faults["timestamp"].max()
    if pd.isna(max_ts):
        return pd.DataFrame(columns=["placa", "alert_type", "severity", "details", "triggered_at"])
    cutoff = max_ts - timedelta(days=float(lookback_days))
    scoped = faults[(faults["diagnostic_id"].isin(event_ids)) & (faults["timestamp"] >= cutoff)]
    if scoped.empty:
        return pd.DataFrame(columns=["placa", "alert_type", "severity", "details", "triggered_at"])
    counts = scoped.groupby("placa").agg(
        eventos=("diagnostic_id", "size"),
        ultimo=("timestamp", "max"),
    )
    min_events = alerts_cfg.get("min_events", 1)
    counts = counts[counts["eventos"] >= int(min_events)]
    if counts.empty:
        return pd.DataFrame(columns=["placa", "alert_type", "severity", "details", "triggered_at"])
    severity = alerts_cfg.get("severity", "medium")
    rows = []
    for placa, row in counts.iterrows():
        detalles = f"{int(row['eventos'])} aceleraciones severas en {int(lookback_days)} días"
        rows.append(
            {
                "placa": placa,
                "alert_type": "driving",
                "severity": severity,
                "details": detalles,
                "triggered_at": row["ultimo"].isoformat(),
            }
        )
    return pd.DataFrame(rows)


def build_pia_alerts(pia_path: Path, config: dict[str, Any]) -> pd.DataFrame:
    try:
        df = pd.read_csv(pia_path)
    except FileNotFoundError:
        return pd.DataFrame()
    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    df["last_record_date"] = pd.to_datetime(df.get("last_record_date"), errors="coerce")
    alerts: list[dict[str, Any]] = []

    alerts_cfg = config.get("alerts", {}) if isinstance(config.get("alerts"), dict) else {}

    downtime_cfg = alerts_cfg.get("downtime", {})
    if isinstance(downtime_cfg, dict):
        threshold = float(downtime_cfg.get("min_hours_30d", 0))
        drop_pct = downtime_cfg.get("min_activity_drop_pct")
        candidates = df[(df.get("downtime_hours_30d", 0) >= threshold) | (df.get("hase_downtime_alert_flag", 0) == 1)]
        if drop_pct is not None:
            candidates = candidates[(candidates.get("activity_drop_pct", 0) >= float(drop_pct)) | (candidates.get("hase_downtime_alert_flag", 0) == 1)]
        for _, row in candidates.iterrows():
            last_ts = row.get("last_record_date")
            if pd.isna(last_ts):
                last_ts = datetime.now(timezone.utc)
            detalle = (
                f"{row.get('downtime_hours_30d', 0):.1f} h sin movimiento en 30d | "
                f"Cobertura 14d: {row.get('coverage_ratio_14d', 0):.2f}"
            )
            alerts.append(
                {
                    "placa": row.get("placa"),
                    "alert_type": "downtime",
                    "severity": downtime_cfg.get("severity", "high"),
                    "details": detalle,
                    "triggered_at": last_ts.isoformat(),
                }
            )

    inactivity_cfg = alerts_cfg.get("inactivity", {})
    if isinstance(inactivity_cfg, dict):
        threshold = float(inactivity_cfg.get("min_activity_drop_pct", 0.4))
        scoped = df[df.get("activity_drop_pct", 0) >= threshold]
        for _, row in scoped.iterrows():
            last_ts = row.get("last_record_date")
            if pd.isna(last_ts):
                last_ts = datetime.now(timezone.utc)
            porcentaje = float(row.get("activity_drop_pct", 0)) * 100
            detalle = f"Actividad cayó {porcentaje:.0f}% vs promedio de 14d"
            alerts.append(
                {
                    "placa": row.get("placa"),
                    "alert_type": "inactivity",
                    "severity": inactivity_cfg.get("severity", "medium"),
                    "details": detalle,
                    "triggered_at": last_ts.isoformat(),
                }
            )

    consumption_cfg = alerts_cfg.get("consumption", {})
    if isinstance(consumption_cfg, dict):
        drop_threshold = float(consumption_cfg.get("drop_pct_threshold", 0.25))
        min_litros = float(consumption_cfg.get("min_litros_30d", 0))
        scoped = df[(df.get("activity_drop_pct", 0) >= drop_threshold) | (df.get("hase_consumption_gap_flag", 0) == 1)]
        scoped = scoped[scoped.get("litros_30d", 0) >= min_litros]
        for _, row in scoped.iterrows():
            last_ts = row.get("last_record_date")
            if pd.isna(last_ts):
                last_ts = datetime.now(timezone.utc)
            drop_pct = float(row.get("activity_drop_pct", 0)) * 100
            consumo = float(row.get("litros_30d", 0))
            detalle = f"Consumo cayó {drop_pct:.0f}% y sólo registra {consumo:,.0f} L en 30d"
            alerts.append(
                {
                    "placa": row.get("placa"),
                    "alert_type": "consumption",
                    "severity": consumption_cfg.get("severity", "medium"),
                    "details": detalle,
                    "triggered_at": last_ts.isoformat(),
                }
            )

    safety_cfg = alerts_cfg.get("safety", {})
    if isinstance(safety_cfg, dict):
        seatbelt_threshold = float(safety_cfg.get("seatbelt_rate_threshold", 0.25))
        speed_threshold = float(safety_cfg.get("high_speed_ratio_threshold", 0.3))
        scoped = df[
            (df.get("safety_alert", 0) == 1)
            | (df.get("seatbelt_off_rate_30d", 0) >= seatbelt_threshold)
            | (df.get("high_speed_ratio_30d", 0) >= speed_threshold)
        ]
        for _, row in scoped.iterrows():
            last_ts = row.get("last_record_date")
            if pd.isna(last_ts):
                last_ts = datetime.now(timezone.utc)
            detalle = (
                f"Seatbelt {row.get('seatbelt_off_rate_30d', 0):.2f} | "
                f"Alta velocidad {row.get('high_speed_ratio_30d', 0):.2f}"
            )
            alerts.append(
                {
                    "placa": row.get("placa"),
                    "alert_type": "safety",
                    "severity": safety_cfg.get("severity", "high"),
                    "details": detalle,
                    "triggered_at": last_ts.isoformat(),
                }
            )

    idle_cfg = alerts_cfg.get("idle", {})
    if isinstance(idle_cfg, dict):
        idle_threshold = float(idle_cfg.get("idle_ratio_threshold", 0.55))
        scoped = df[df.get("idle_hours_ratio_30d", 0) >= idle_threshold]
        for _, row in scoped.iterrows():
            last_ts = row.get("last_record_date")
            if pd.isna(last_ts):
                last_ts = datetime.now(timezone.utc)
            detalle = f"Ralenti {row.get('idle_hours_ratio_30d', 0):.2f} de horas activas"
            alerts.append(
                {
                    "placa": row.get("placa"),
                    "alert_type": "idle",
                    "severity": idle_cfg.get("severity", "medium"),
                    "details": detalle,
                    "triggered_at": last_ts.isoformat(),
                }
            )

    after_hours_cfg = alerts_cfg.get("after_hours", {})
    if isinstance(after_hours_cfg, dict):
        ratio_threshold = float(after_hours_cfg.get("ratio_threshold", 0.35))
        km_threshold = float(after_hours_cfg.get("min_km_30d", 100))
        scoped = df[
            (df.get("after_hours_ratio_30d", 0) >= ratio_threshold)
            & (df.get("after_hours_distance_km_30d", 0) >= km_threshold)
        ]
        for _, row in scoped.iterrows():
            last_ts = row.get("last_record_date")
            if pd.isna(last_ts):
                last_ts = datetime.now(timezone.utc)
            detalle = (
                f"Uso fuera de horario {row.get('after_hours_ratio_30d', 0):.2f} del total | "
                f"Distancia: {row.get('after_hours_distance_km_30d', 0):.0f} km"
            )
            alerts.append(
                {
                    "placa": row.get("placa"),
                    "alert_type": "after_hours",
                    "severity": after_hours_cfg.get("severity", "medium"),
                    "details": detalle,
                    "triggered_at": last_ts.isoformat(),
                }
            )

    telem_cfg = alerts_cfg.get("telemetry", {})
    if isinstance(telem_cfg, dict):
        health_threshold = float(telem_cfg.get("max_health_score", 0.45))
        scoped = df[df.get("telemetry_health_score", 1) <= health_threshold]
        for _, row in scoped.iterrows():
            last_ts = row.get("last_record_date")
            if pd.isna(last_ts):
                last_ts = datetime.now(timezone.utc)
            detalle = f"Salud de telemetría {row.get('telemetry_health_score', 0):.2f}"
            alerts.append(
                {
                    "placa": row.get("placa"),
                    "alert_type": "telemetry",
                    "severity": telem_cfg.get("severity", "high"),
                    "details": detalle,
                    "triggered_at": last_ts.isoformat(),
                }
            )

    return pd.DataFrame(alerts)


def load_events_data(events_daily_path: Path, events_percentiles_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    events_daily = pd.DataFrame()
    events_percentiles = pd.DataFrame()

    if events_daily_path and events_daily_path.exists():
        try:
            events_daily = pd.read_csv(events_daily_path)
        except FileNotFoundError:
            pass

    if events_percentiles_path and events_percentiles_path.exists():
        try:
            events_percentiles = pd.read_csv(events_percentiles_path)
        except FileNotFoundError:
            pass

    return events_daily, events_percentiles


def build_enhanced_safety_alerts(events_daily: pd.DataFrame, events_percentiles: pd.DataFrame, config: dict[str, Any], device_map: dict[str, str]) -> pd.DataFrame:
    if events_daily.empty:
        return pd.DataFrame(columns=["placa", "alert_type", "severity", "details", "triggered_at"])

    alerts = []
    safety_cfg = config.get("alerts", {}).get("enhanced_safety", {})

    if not isinstance(safety_cfg, dict):
        return pd.DataFrame(columns=["placa", "alert_type", "severity", "details", "triggered_at"])

    # Agregar placa si no existe
    if "placa" not in events_daily.columns and "device" in events_daily.columns:
        events_daily["placa"] = events_daily["device"].map(device_map)

    # Frenado brusco frecuente
    harsh_braking_threshold = safety_cfg.get("harsh_braking_rate_threshold", 5)  # Ajustado a eventos absolutos
    harsh_braking_events = events_daily[events_daily.get("harsh_brake", 0) > harsh_braking_threshold]

    for _, row in harsh_braking_events.iterrows():
        harsh_brake_count = row.get("harsh_brake", 0)
        alerts.append({
            "placa": row.get("placa"),
            "alert_type": "enhanced_safety",
            "severity": safety_cfg.get("severity", "high"),
            "details": f"Frenado brusco frecuente: {harsh_brake_count} eventos (>{harsh_braking_threshold})",
            "triggered_at": row.get("fecha", datetime.now().strftime("%Y-%m-%d"))
        })

    # Violaciones de cinturón
    seatbelt_threshold = safety_cfg.get("seatbelt_violations_per_hour", 3)
    seatbelt_violations = events_daily[events_daily.get("seatbelt_off", 0) > seatbelt_threshold]

    for _, row in seatbelt_violations.iterrows():
        seatbelt_events = row.get("seatbelt_off", 0)
        alerts.append({
            "placa": row.get("placa"),
            "alert_type": "enhanced_safety",
            "severity": safety_cfg.get("severity", "high"),
            "details": f"Violaciones cinturón: {seatbelt_events} eventos (>{seatbelt_threshold})",
            "triggered_at": row.get("fecha", datetime.now().strftime("%Y-%m-%d"))
        })

    # Maniobras bruscas
    maneuver_threshold = safety_cfg.get("harsh_maneuver_threshold", 50)
    harsh_maneuvers = events_daily[events_daily.get("harsh_maneuver", 0) > maneuver_threshold]

    for _, row in harsh_maneuvers.iterrows():
        maneuver_events = row.get("harsh_maneuver", 0)
        alerts.append({
            "placa": row.get("placa"),
            "alert_type": "enhanced_safety",
            "severity": safety_cfg.get("severity", "high"),
            "details": f"Maniobras bruscas: {maneuver_events} eventos (>{maneuver_threshold})",
            "triggered_at": row.get("fecha", datetime.now().strftime("%Y-%m-%d"))
        })

    # Exceso de velocidad
    overspeed_threshold = safety_cfg.get("overspeed_threshold", 20)
    overspeed_events = events_daily[events_daily.get("overspeed", 0) > overspeed_threshold]

    for _, row in overspeed_events.iterrows():
        speed_events = row.get("overspeed", 0)
        alerts.append({
            "placa": row.get("placa"),
            "alert_type": "enhanced_safety",
            "severity": safety_cfg.get("severity", "high"),
            "details": f"Exceso de velocidad: {speed_events} eventos (>{overspeed_threshold})",
            "triggered_at": row.get("fecha", datetime.now().strftime("%Y-%m-%d"))
        })

    return pd.DataFrame(alerts)


def build_enhanced_operations_alerts(events_daily: pd.DataFrame, config: dict[str, Any], device_map: dict[str, str]) -> pd.DataFrame:
    if events_daily.empty:
        return pd.DataFrame(columns=["placa", "alert_type", "severity", "details", "triggered_at"])

    alerts = []
    ops_cfg = config.get("alerts", {}).get("enhanced_operations", {})

    if not isinstance(ops_cfg, dict):
        return pd.DataFrame(columns=["placa", "alert_type", "severity", "details", "triggered_at"])

    # Agregar placa si no existe
    if "placa" not in events_daily.columns and "device" in events_daily.columns:
        events_daily["placa"] = events_daily["device"].map(device_map)

    # Ralentí excesivo
    idle_threshold = ops_cfg.get("idle_engine_threshold_minutes", 30)  # Ajustado a eventos
    idle_violations = events_daily[events_daily.get("idling", 0) > idle_threshold]

    for _, row in idle_violations.iterrows():
        idle_events = row.get("idling", 0)
        alerts.append({
            "placa": row.get("placa"),
            "alert_type": "enhanced_operations",
            "severity": ops_cfg.get("severity", "medium"),
            "details": f"Ralentí excesivo: {idle_events} eventos (>{idle_threshold})",
            "triggered_at": row.get("fecha", datetime.now().strftime("%Y-%m-%d"))
        })

    # PTO no autorizado
    pto_threshold = ops_cfg.get("pto_unauthorized_threshold_minutes", 5)
    pto_violations = events_daily[events_daily.get("pto", 0) > pto_threshold]

    for _, row in pto_violations.iterrows():
        pto_events = row.get("pto", 0)
        alerts.append({
            "placa": row.get("placa"),
            "alert_type": "enhanced_operations",
            "severity": ops_cfg.get("severity", "medium"),
            "details": f"PTO activo: {pto_events} eventos (>{pto_threshold})",
            "triggered_at": row.get("fecha", datetime.now().strftime("%Y-%m-%d"))
        })

    return pd.DataFrame(alerts)


def combine_alerts(frames: list[pd.DataFrame]) -> pd.DataFrame:
    frames = [df for df in frames if not df.empty]
    if not frames:
        return pd.DataFrame(columns=["placa", "alert_type", "severity", "details", "triggered_at"])
    combined = pd.concat(frames, ignore_index=True)
    severity_order = {"high": 0, "medium": 1, "low": 2, "info": 3}
    combined["_severity_order"] = combined["severity"].map(lambda s: severity_order.get(s, 4))
    combined = combined.sort_values(["_severity_order", "triggered_at", "placa"])
    combined = combined.drop(columns=["_severity_order"], errors="ignore")
    combined = combined.drop_duplicates(subset=["placa", "alert_type", "details"], keep="first")
    return combined


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_yaml(args.config)
    if args.output is not None:
        config.setdefault("paths", {})["output"] = str(args.output)
    paths = config.get("paths", {})
    devices_path = (ROOT / paths["devices"]) if paths.get("devices") else None
    faults_path = (ROOT / paths["faults"]) if paths.get("faults") else None
    pia_features_path = (ROOT / paths["pia_features"]) if paths.get("pia_features") else None
    events_daily_path = (ROOT / paths["events_daily"]) if paths.get("events_daily") else None
    events_percentiles_path = (ROOT / paths["events_percentiles"]) if paths.get("events_percentiles") else None
    output_path = ROOT / paths.get("output", "data/guardian/guardian_insights.csv")

    device_map = load_device_map(devices_path) if devices_path else {}
    dtc_catalog = load_dtc_catalog()
    faults_df = load_faults(faults_path, device_map) if faults_path else pd.DataFrame()
    events_daily, events_percentiles = load_events_data(events_daily_path, events_percentiles_path)

    dtc_alerts = build_dtc_alerts(faults_df, config, dtc_catalog)
    driving_alerts = build_driving_alerts(faults_df, config)
    pia_alerts = build_pia_alerts(pia_features_path, config) if pia_features_path else pd.DataFrame()
    enhanced_safety_alerts = build_enhanced_safety_alerts(events_daily, events_percentiles, config, device_map)
    enhanced_operations_alerts = build_enhanced_operations_alerts(events_daily, config, device_map)

    combined = combine_alerts([dtc_alerts, driving_alerts, pia_alerts, enhanced_safety_alerts, enhanced_operations_alerts])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)

    inputs = [Path(args.config)]
    for candidate in (devices_path, faults_path, pia_features_path, events_daily_path, events_percentiles_path):
        if candidate:
            inputs.append(candidate)
    for catalog in (DTC_CATALOG_PATH, GLOBAL_DTC_PATH):
        if catalog.exists():
            inputs.append(catalog)

    if combined.empty:
        print(f"Guardian no encontró alertas con los filtros actuales → {_rel(output_path)}")
        write_metadata(
            output_path,
            script=__file__,
            inputs=inputs,
            extra={"rows": 0},
        )
    else:
        resumen = combined.groupby("alert_type").size().to_dict()
        print(f"Guardian generó {len(combined)} alertas → {_rel(output_path)}")
        for alert_type, count in resumen.items():
            print(f"  - {alert_type}: {count}")
        write_metadata(
            output_path,
            script=__file__,
            inputs=inputs,
            extra={"rows": int(len(combined)), "by_type": resumen},
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
