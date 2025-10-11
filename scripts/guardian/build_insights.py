#!/usr/bin/env python3
"""Construye `data/guardian_insights.csv` combinando señales de telemetría."""

from __future__ import annotations

import argparse
import ast
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config" / "guardian.yml"
DTC_CATALOG_PATH = ROOT / "guardian" / "dtc_catalog.json"
GLOBAL_DTC_PATH = ROOT / "data" / "dtc_catalog.json"

Severity = str


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
    mapping: dict[str, str] = {}
    for _, row in df.iterrows():
        device_id = row.get("id")
        plate = row.get("licensePlate") or row.get("name")
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
    df["device_id"] = df["device"].apply(normalize_device_id)
    df["placa"] = df["device_id"].map(device_map)
    df["diagnostic_id"] = df["diagnostic"].apply(extract_diagnostic_id)
    df["timestamp"] = pd.to_datetime(df["dateTime"], errors="coerce")
    df = df.dropna(subset=["placa", "diagnostic_id", "timestamp"])
    return df


def _resolve_dtc_severity(row: pd.Series, config: dict[str, Any], entry: dict[str, Any]) -> Severity:
    cfg = config.get("alerts", {}).get("dtc", {})
    severity_map = cfg.get("severity_map", {}) if isinstance(cfg, dict) else {}
    fallback = cfg.get("fallback_severity", "medium") if isinstance(cfg, dict) else "medium"
    candidate: str | None = None
    diag_sev = row.get("diagnosticSeverity")
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
        fault_state = row.get("faultState")
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

    downtime_cfg = config.get("alerts", {}).get("downtime", {})
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

    inactivity_cfg = config.get("alerts", {}).get("inactivity", {})
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

    consumption_cfg = config.get("alerts", {}).get("consumption", {})
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
    output_path = ROOT / paths.get("output", "data/guardian/guardian_insights.csv")

    device_map = load_device_map(devices_path) if devices_path else {}
    dtc_catalog = load_dtc_catalog()
    faults_df = load_faults(faults_path, device_map) if faults_path else pd.DataFrame()
    dtc_alerts = build_dtc_alerts(faults_df, config, dtc_catalog)
    driving_alerts = build_driving_alerts(faults_df, config)
    pia_alerts = build_pia_alerts(pia_features_path, config) if pia_features_path else pd.DataFrame()

    combined = combine_alerts([dtc_alerts, driving_alerts, pia_alerts])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)

    if combined.empty:
        print(f"Guardian no encontró alertas con los filtros actuales → {output_path}")
    else:
        resumen = combined.groupby("alert_type").size().to_dict()
        print(f"Guardian generó {len(combined)} alertas → {output_path}")
        for alert_type, count in resumen.items():
            print(f"  - {alert_type}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
