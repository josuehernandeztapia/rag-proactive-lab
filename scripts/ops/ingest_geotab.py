#!/usr/bin/env python3
"""Normaliza los CSV exportados de Geotab y genera staging datasets compartidos.

Usage:
    python scripts/ops/ingest_geotab.py --source data/raw/geotab
    python scripts/ops/ingest_geotab.py \
        --devices data/raw/geotab/Device.csv \
        --faults data/raw/geotab/FaultData.csv \
        --trips data/raw/geotab/Trip.csv \
        --rules data/raw/geotab/Rule.csv \
        --zones data/raw/geotab/Zone.csv \
        --logrecord data/raw/geotab/LogRecord.csv \
        --status data/raw/geotab/StatusData.csv

Los archivos normalizados se escriben en `data/staging/` y alimentan a HASE, PIA y Guardian."""

from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.loader import PROJECT_ROOT, get_path
from config.metadata import write_metadata

DEFAULT_SOURCE = get_path('data', 'raw', 'geotab_dir')
STAGING_DIR = get_path('data', 'staging', 'root')
GEOTAB_DEVICES_OUT = get_path('data', 'staging', 'geotab_devices')
GEOTAB_FAULTS_OUT = get_path('data', 'staging', 'geotab_faults')
GEOTAB_TRIPS_DAILY_OUT = get_path('data', 'staging', 'geotab_trips_daily')
GEOTAB_TRIPS_RAW_OUT = get_path('data', 'staging', 'geotab_trips_raw')
GEOTAB_RULES_OUT = get_path('data', 'staging', 'geotab_rules')
GEOTAB_ZONES_OUT = get_path('data', 'staging', 'geotab_zones')
TELEMETRY_SUMMARY_OUT = get_path('data', 'staging', 'telemetry_summary')
GEOTAB_EVENTS_RAW_OUT = get_path('data', 'staging', 'geotab_events_raw', default=STAGING_DIR / 'geotab_events_raw.csv', required=False)
GEOTAB_EVENTS_DAILY_OUT = get_path('data', 'staging', 'geotab_events_daily', default=STAGING_DIR / 'geotab_events_daily.csv', required=False)
GEOTAB_EVENTS_MARKET_PCTL_OUT = get_path('data', 'staging', 'geotab_events_market_percentiles', default=STAGING_DIR / 'geotab_events_market_percentiles.csv', required=False)


@dataclass
class Paths:
    devices: Path
    faults: Path
    trips: Path
    rules: Path
    zones: Path
    users: Path | None = None
    logrecord: Path | None = None
    status: Path | None = None


@dataclass
class Outputs:
    devices: Path
    faults: Path
    trips_daily: Path
    trips_raw: Path
    rules: Path
    zones: Path
    telemetry_summary: Path
    events_raw: Path | None
    events_daily: Path | None
    events_percentiles: Path | None


DICT_PREFIX = ("{\"id\"", "{'id':")


def _parse_dict(value: Any) -> dict[str, Any] | None:
    if isinstance(value, str) and value.strip().startswith(DICT_PREFIX):
        try:
            return ast.literal_eval(value)
        except Exception:
            return None
    if isinstance(value, dict):
        return value
    return None


def _parse_market(value: Any) -> str | None:
    parsed = _parse_dict(value)
    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, dict):
                name = str(item.get('name', '')).strip()
                if name:
                    lowered = name.lower()
                    if 'market' in lowered or 'mercado' in lowered:
                        return name
        for item in parsed:
            if isinstance(item, dict):
                name = str(item.get('name', '')).strip()
                if name:
                    return name
    if isinstance(parsed, dict):
        name = str(parsed.get('name', '')).strip()
        return name or None
    if isinstance(value, str):
        lowered = value.lower()
        if 'market' in lowered or 'mercado' in lowered:
            return value
    return None


def _ensure_output_dirs() -> None:
    STAGING_DIR.mkdir(parents=True, exist_ok=True)


def _with_suffix(path: Path | None, suffix: str | None) -> Path | None:
    if path is None or not suffix:
        return path
    return path.with_name(f"{path.stem}_{suffix}{path.suffix}")


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def load_devices(path: Path, out_path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    columns = {
        "id": "device_id",
        "licensePlate": "license_plate",
        "name": "device_name",
        "serialNumber": "serial_number",
        "groups": "groups_raw",
        "comment": "comment",
        "timeZoneId": "timezone_id",
        "vehicleIdentificationNumber": "vin",
    }
    subset = df[list(columns.keys())].rename(columns=columns)
    subset["market"] = subset["groups_raw"].apply(_parse_market).fillna("unknown")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(out_path, index=False)
    write_metadata(
        out_path,
        script=__file__,
        inputs=[path],
        extra={'rows': int(len(subset))}
    )
    return subset


def normalize_faults(path: Path, devices: pd.DataFrame, out_path: Path, devices_out: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["device_id"] = df["device"].apply(lambda x: (_parse_dict(x) or {}).get("id"))
    df["diagnostic_id"] = df["diagnostic"].apply(lambda x: (_parse_dict(x) or {}).get("id"))
    df["controller_id"] = df["controller"].apply(lambda x: (_parse_dict(x) or {}).get("id"))
    df["effective_status"] = df["faultStates"].apply(
        lambda x: (_parse_dict(x) or {}).get("effectiveStatus") if isinstance(x, (str, dict)) else None
    )
    df["timestamp"] = pd.to_datetime(df["dateTime"], errors="coerce")
    merged = df.merge(devices[["device_id", "license_plate"]], on="device_id", how="left")
    keep_cols = [
        "timestamp",
        "license_plate",
        "device_id",
        "diagnostic_id",
        "diagnosticSeverity",
        "failureMode",
        "faultState",
        "effective_status",
        "malfunctionLamp",
        "amberWarningLamp",
        "redStopLamp",
        "protectWarningLamp",
        "riskOfBreakdown",
        "severity",
        "recommendation",
        "count",
        "sourceAddress",
    ]
    normalized = merged[keep_cols].rename(
        columns={
            "diagnosticSeverity": "diagnostic_severity",
            "failureMode": "failure_mode",
            "faultState": "fault_state",
            "malfunctionLamp": "malfunction_lamp",
            "amberWarningLamp": "amber_warning_lamp",
            "redStopLamp": "red_stop_lamp",
            "protectWarningLamp": "protect_warning_lamp",
        }
    )
    normalized.sort_values("timestamp", inplace=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(out_path, index=False)
    write_metadata(
        out_path,
        script=__file__,
        inputs=[path, devices_out],
        extra={'rows': int(len(normalized))}
    )
    return normalized


def _to_seconds(series: pd.Series) -> pd.Series:
    return pd.to_timedelta(series, errors="coerce").dt.total_seconds()


def normalize_trips(path: Path, devices: pd.DataFrame, out_daily: Path, out_raw: Path, devices_out: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["device_id"] = df["device"].apply(lambda x: (_parse_dict(x) or {}).get("id"))
    df = df.merge(devices[["device_id", "license_plate"]], on="device_id", how="left")
    for col in [
        "drivingDuration",
        "engineHours",
        "idlingDuration",
        "afterHoursDrivingDuration",
        "afterHoursStopDuration",
        "workDrivingDuration",
        "workStopDuration",
    ]:
        df[f"{col}_sec"] = _to_seconds(df[col])
    for col in ["speedRange1Duration", "speedRange2Duration", "speedRange3Duration", "stopDuration"]:
        df[f"{col}_sec"] = _to_seconds(df[col])
    numeric_cols = [
        "distance",
        "workDistance",
        "afterHoursDistance",
        "maximumSpeed",
        "speedRange1",
        "speedRange2",
        "speedRange3",
        "averageSpeed",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["isSeatBeltOff"] = df["isSeatBeltOff"].astype(bool).astype(int)
    df["start"] = pd.to_datetime(df["start"], errors="coerce")
    df["date"] = df["start"].dt.date
    grouped = (
        df.groupby(["license_plate", "date"], dropna=False)
        .agg(
            distance=("distance", "sum"),
            workDistance=("workDistance", "sum"),
            afterHoursDistance=("afterHoursDistance", "sum"),
            maximumSpeed=("maximumSpeed", "max"),
            drivingDuration_sec=("drivingDuration_sec", "sum"),
            engineHours_sec=("engineHours_sec", "sum"),
            idlingDuration_sec=("idlingDuration_sec", "sum"),
            afterHoursDrivingDuration_sec=("afterHoursDrivingDuration_sec", "sum"),
            afterHoursStopDuration_sec=("afterHoursStopDuration_sec", "sum"),
            workDrivingDuration_sec=("workDrivingDuration_sec", "sum"),
            workStopDuration_sec=("workStopDuration_sec", "sum"),
            averageSpeed=("averageSpeed", "mean"),
            speedRange1=("speedRange1", "sum"),
            speedRange2=("speedRange2", "sum"),
            speedRange3=("speedRange3", "sum"),
            speedRange1Duration_sec=("speedRange1Duration_sec", "sum"),
            speedRange2Duration_sec=("speedRange2Duration_sec", "sum"),
            speedRange3Duration_sec=("speedRange3Duration_sec", "sum"),
            stopDuration_sec=("stopDuration_sec", "sum"),
            seatbelt_off_events=("isSeatBeltOff", "sum"),
            trip_count=("distance", "size"),
        )
        .reset_index()
        .rename(columns={"license_plate": "placa", "date": "fecha"})
    )
    grouped["distance_km"] = grouped["distance"].fillna(0) / 1000.0
    grouped["driving_hours"] = grouped["drivingDuration_sec"].fillna(0) / 3600.0
    grouped["engine_hours"] = grouped["engineHours_sec"].fillna(0) / 3600.0
    grouped["idling_hours"] = grouped["idlingDuration_sec"].fillna(0) / 3600.0
    grouped["after_hours_distance_km"] = grouped["afterHoursDistance"].fillna(0) / 1000.0
    grouped["after_hours_driving_hours"] = grouped["afterHoursDrivingDuration_sec"].fillna(0) / 3600.0
    grouped["after_hours_stop_hours"] = grouped["afterHoursStopDuration_sec"].fillna(0) / 3600.0
    grouped["work_distance_km"] = grouped["workDistance"].fillna(0) / 1000.0
    grouped["work_driving_hours"] = grouped["workDrivingDuration_sec"].fillna(0) / 3600.0
    grouped["work_stop_hours"] = grouped["workStopDuration_sec"].fillna(0) / 3600.0
    grouped["max_speed_kph"] = grouped["maximumSpeed"].fillna(0)
    grouped["average_speed_kph"] = grouped["averageSpeed"].fillna(0)
    grouped["speed_range1_km"] = grouped["speedRange1"].fillna(0)
    grouped["speed_range2_km"] = grouped["speedRange2"].fillna(0)
    grouped["speed_range3_km"] = grouped["speedRange3"].fillna(0)
    grouped["speed_range1_hours"] = grouped["speedRange1Duration_sec"].fillna(0) / 3600.0
    grouped["speed_range2_hours"] = grouped["speedRange2Duration_sec"].fillna(0) / 3600.0
    grouped["speed_range3_hours"] = grouped["speedRange3Duration_sec"].fillna(0) / 3600.0
    grouped["stop_hours"] = grouped["stopDuration_sec"].fillna(0) / 3600.0
    grouped["seatbelt_off_events"] = grouped["seatbelt_off_events"].fillna(0).astype(int)
    grouped["trip_count"] = grouped["trip_count"].fillna(0).astype(int)

    grouped.drop(
        columns=[
            "averageSpeed",
            "speedRange1",
            "speedRange2",
            "speedRange3",
            "speedRange1Duration_sec",
            "speedRange2Duration_sec",
            "speedRange3Duration_sec",
            "stopDuration_sec",
        ],
        inplace=True,
        errors="ignore",
    )
    out_daily.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(out_daily, index=False)

    write_metadata(
        out_daily,
        script=__file__,
        inputs=[path, devices_out],
        extra={'rows': int(len(grouped))}
    )

    keep_raw_cols = [
        "device_id",
        "license_plate",
        "start",
        "stop",
        "distance",
        "drivingDuration",
        "engineHours",
        "idlingDuration",
        "maximumSpeed",
        "afterHoursDistance",
        "afterHoursDrivingDuration",
        "afterHoursStopDuration",
        "workDistance",
        "workDrivingDuration",
        "workStopDuration",
    ]
    out_raw.parent.mkdir(parents=True, exist_ok=True)
    df[keep_raw_cols].to_csv(out_raw, index=False)
    write_metadata(
        out_raw,
        script=__file__,
        inputs=[path],
        extra={'rows': int(len(df))}
    )
    return grouped


def _coerce_timestamp(df: pd.DataFrame, *candidates: str) -> pd.Series:
    for col in candidates:
        if col in df:
            ts = pd.to_datetime(df[col], errors='coerce')
            if ts.notna().any():
                return ts
    return pd.Series([pd.NaT] * len(df))


def _event_type_from_rule(rule_info: dict[str, Any]) -> str:
    name = str(rule_info.get('name', '')).lower()
    if not name:
        return 'other'
    if 'brake' in name or 'fren' in name:
        return 'harsh_brake'
    if 'seat' in name and 'belt' in name:
        return 'seatbelt_off'
    if 'speed' in name or 'velocidad' in name:
        return 'overspeed'
    if 'pto' in name:
        return 'pto'
    if 'after hour' in name or 'fuera de horario' in name:
        return 'after_hours'
    if 'idle' in name or 'ralent' in name:
        return 'idling'
    if 'disconnect' in name or 'desconex' in name:
        return 'device_disconnect'
    return 'other'


def _event_type_from_diagnostic(diag_info: dict[str, Any], value: Any) -> str:
    diag_id = str(diag_info.get('id', ''))
    name = str(diag_info.get('name', '')).lower()

    # Clasificación basada en IDs específicos de Geotab
    if diag_id in ['DiagnosticIgnitionId', 'DiagnosticVehicleActiveId']:
        try:
            numeric = float(value)
            return 'ignition_on' if numeric > 0 else 'ignition_off'
        except Exception:
            return 'other'

    # Eventos de frenado y aceleración brusca
    if diag_id == 'DiagnosticAccelerationForwardBrakingId':
        try:
            numeric = float(value)
            # Umbral para detectar frenado brusco (valores negativos altos)
            return 'harsh_brake' if numeric < -0.3 else 'other'
        except Exception:
            return 'other'

    # Aceleración lateral y vertical (maniobras bruscas)
    if diag_id in ['DiagnosticAccelerationUpDownId', 'DiagnosticAccelerationSideToSideId']:
        try:
            numeric = abs(float(value))
            # Umbral para detectar maniobras bruscas
            return 'harsh_maneuver' if numeric > 0.4 else 'other'
        except Exception:
            return 'other'

    # Velocidad del vehículo
    if diag_id == 'DiagnosticEngineRoadSpeedId':
        try:
            numeric = float(value)
            # Umbral para detectar exceso de velocidad (>80 km/h)
            return 'overspeed' if numeric > 80 else 'other'
        except Exception:
            return 'other'

    # Velocidad del motor (para detectar ralentí)
    if diag_id == 'DiagnosticEngineSpeedId':
        try:
            numeric = float(value)
            # RPM entre 600-1000 indica ralentí
            return 'idling' if 600 <= numeric <= 1000 else 'other'
        except Exception:
            return 'other'

    # Voltaje del dispositivo (para detectar desconexiones)
    if diag_id in ['DiagnosticGoDeviceVoltageId', 'DiagnosticCrankingVoltageId']:
        try:
            numeric = float(value)
            # Voltaje bajo indica posible desconexión
            return 'device_disconnect' if numeric < 10 else 'other'
        except Exception:
            return 'other'

    # Fallback a clasificación por nombre si no hay ID específico
    if 'ignition' in name or 'ignición' in name:
        try:
            numeric = float(value)
            return 'ignition_on' if numeric > 0 else 'ignition_off'
        except Exception:
            return 'other'

    if 'seat' in name and 'belt' in name:
        try:
            numeric = float(value)
            return 'seatbelt_off' if numeric == 0 else 'other'
        except Exception:
            return 'other'

    if 'pto' in name:
        return 'pto'

    return 'other'


def load_log_records(path: Path, devices: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['device_id'] = df['device'].apply(lambda x: (_parse_dict(x) or {}).get('id'))
    df = df.merge(devices[["device_id", "license_plate", "market"]], on="device_id", how="left")
    df["timestamp"] = _coerce_timestamp(df, 'dateTime', 'dateTimeUTC', 'time')
    df = df.dropna(subset=["timestamp", "license_plate"])
    df['event_type'] = df.get('rule', pd.Series([None] * len(df))).apply(
        lambda x: _event_type_from_rule(_parse_dict(x) or {})
    )
    df['value'] = df.get('duration', df.get('length', pd.Series([None] * len(df))))
    events = df[["timestamp", "license_plate", "market", "event_type", "value"]].rename(
        columns={"license_plate": "placa"}
    )
    return events


def load_status_data(path: Path, devices: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['device_id'] = df['device'].apply(lambda x: (_parse_dict(x) or {}).get('id'))
    df = df.merge(devices[["device_id", "license_plate", "market"]], on="device_id", how="left")
    df["timestamp"] = _coerce_timestamp(df, 'dateTime', 'dateTimeUTC', 'time')
    df = df.dropna(subset=["timestamp", "license_plate"])
    values = df.get('data') if 'data' in df else pd.Series([None] * len(df))
    diagnostics = df.get('diagnostic', pd.Series([None] * len(df)))
    df['event_type'] = [
        _event_type_from_diagnostic(_parse_dict(diag) or {}, val)
        for diag, val in zip(diagnostics, values)
    ]
    df['value'] = values
    events = df[["timestamp", "license_plate", "market", "event_type", "value"]].rename(
        columns={"license_plate": "placa"}
    )
    return events


def aggregate_events_daily(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    events['fecha'] = events['timestamp'].dt.date
    expected = [
        'harsh_brake', 'harsh_maneuver', 'seatbelt_off', 'overspeed', 'idling', 'pto',
        'ignition_on', 'ignition_off', 'after_hours', 'device_disconnect', 'other'
    ]
    grouped = (
        events.groupby(['placa', 'market', 'fecha'])['event_type']
        .value_counts()
        .unstack(fill_value=0)
        .reset_index()
    )
    for name in expected:
        if name not in grouped:
            grouped[name] = 0
    ordered_cols = ['placa', 'market', 'fecha', *expected]
    grouped = grouped[ordered_cols]
    grouped['total_events'] = grouped[expected].sum(axis=1)
    return grouped


def compute_market_percentiles(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame()
    metrics = [
        'harsh_brake', 'harsh_maneuver', 'seatbelt_off', 'overspeed', 'idling', 'pto',
        'ignition_on', 'ignition_off', 'after_hours', 'device_disconnect', 'other', 'total_events'
    ]
    records = []
    for market, group in daily.groupby('market'):
        for metric in metrics:
            series = group.get(metric)
            if series is None:
                continue
            records.append({
                'market': market,
                'metric': metric,
                'p95': float(series.quantile(0.95)),
                'p99': float(series.quantile(0.99)),
                'mean': float(series.mean()),
            })
    return pd.DataFrame(records)


def ingest_events(logrecord_path: Path | None, status_path: Path | None, devices: pd.DataFrame, out_raw: Path | None, out_daily: Path | None, out_pct: Path | None) -> None:
    frames: list[pd.DataFrame] = []
    inputs: list[Path] = []

    if logrecord_path and logrecord_path.exists():
        frames.append(load_log_records(logrecord_path, devices))
        inputs.append(logrecord_path)
    elif logrecord_path:
        print(f"[geotab] LogRecord no encontrado en {logrecord_path}")

    if status_path and status_path.exists():
        frames.append(load_status_data(status_path, devices))
        inputs.append(status_path)
    elif status_path:
        print(f"[geotab] StatusData no encontrado en {status_path}")

    if not frames:
        print("[geotab] Eventos crudos omitidos (no se encontraron LogRecord/StatusData)")
        return

    if out_raw is None or out_daily is None or out_pct is None:
        print("[geotab] Ruta de salida para eventos no configurada; se omite escritura")
        return

    events = pd.concat(frames, ignore_index=True)
    events.sort_values("timestamp", inplace=True)
    out_raw.parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(out_raw, index=False)
    write_metadata(
        out_raw,
        script=__file__,
        inputs=inputs,
        extra={'rows': int(len(events))}
    )
    print(f"[geotab] Eventos crudos: {len(events)} filas → {_rel(out_raw)}")

    daily = aggregate_events_daily(events)
    out_daily.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(out_daily, index=False)
    write_metadata(
        out_daily,
        script=__file__,
        inputs=[out_raw],
        extra={'rows': int(len(daily))}
    )
    print(f"[geotab] Eventos diarios: {len(daily)} filas → {_rel(out_daily)}")

    percentiles = compute_market_percentiles(daily)
    out_pct.parent.mkdir(parents=True, exist_ok=True)
    percentiles.to_csv(out_pct, index=False)
    write_metadata(
        out_pct,
        script=__file__,
        inputs=[out_daily],
        extra={'rows': int(len(percentiles))}
    )
    print(f"[geotab] Percentiles por mercado → {_rel(out_pct)}")


def create_telemetry_summary(trip_daily: pd.DataFrame, out_path: Path, trips_daily_out: Path) -> pd.DataFrame:
    telemetry = trip_daily.copy()
    telemetry.rename(columns={"fecha": "fecha_dia", "placa": "placa"}, inplace=True)
    telemetry["downtime_hours"] = 24 - telemetry["driving_hours"].fillna(0) - telemetry["idling_hours"].fillna(0)
    telemetry.loc[telemetry["downtime_hours"] < 0, "downtime_hours"] = 0
    telemetry.sort_values(["placa", "fecha_dia"], inplace=True)
    telemetry["fecha_dia"] = pd.to_datetime(telemetry["fecha_dia"])
    telemetry["driving_seconds"] = telemetry["driving_hours"] * 3600
    telemetry["rolling_mean"] = (
        telemetry.groupby("placa")["driving_seconds"].rolling(window=14, min_periods=3).mean().shift(1).values
    )
    telemetry["activity_drop_pct"] = np.where(
        telemetry["rolling_mean"] > 0,
        np.clip(1 - (telemetry["driving_seconds"] / telemetry["rolling_mean"]), 0, 1),
        0,
    )
    enriched_cols = [
        "average_speed_kph",
        "trip_count",
        "stop_hours",
        "seatbelt_off_events",
        "speed_range1_hours",
        "speed_range2_hours",
        "speed_range3_hours",
        "speed_range1_km",
        "speed_range2_km",
        "speed_range3_km",
        "max_speed_kph",
        "driving_hours",
        "engine_hours",
        "idling_hours",
        "after_hours_distance_km",
        "after_hours_driving_hours",
        "after_hours_stop_hours",
        "work_distance_km",
        "work_driving_hours",
        "work_stop_hours",
    ]
    for col in enriched_cols:
        if col in telemetry.columns:
            telemetry[col] = telemetry[col].fillna(0)
        else:
            telemetry[col] = 0
    summary = telemetry[[
        "placa",
        "fecha_dia",
        "downtime_hours",
        "activity_drop_pct",
        *enriched_cols,
    ]]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)
    write_metadata(
        out_path,
        script=__file__,
        inputs=[trips_daily_out],
        extra={'rows': int(len(summary))}
    )
    return summary


def save_rules_zones(rules_path: Path, zones_path: Path, out_rules: Path, out_zones: Path) -> None:
    out_rules.parent.mkdir(parents=True, exist_ok=True)
    out_zones.parent.mkdir(parents=True, exist_ok=True)
    pd.read_csv(rules_path).to_csv(out_rules, index=False)
    pd.read_csv(zones_path).to_csv(out_zones, index=False)
    write_metadata(out_rules, script=__file__, inputs=[rules_path])
    write_metadata(out_zones, script=__file__, inputs=[zones_path])


def ingest(paths: Paths, outputs: Outputs) -> None:
    _ensure_output_dirs()
    devices = load_devices(paths.devices, outputs.devices)
    print(f"[geotab] Dispositivos normalizados → {_rel(outputs.devices)}")
    faults = normalize_faults(paths.faults, devices, outputs.faults, outputs.devices)
    print(f"[geotab] Faults normalizados: {len(faults)} filas → {_rel(outputs.faults)}")
    trips = normalize_trips(paths.trips, devices, outputs.trips_daily, outputs.trips_raw, outputs.devices)
    print(f"[geotab] Trips agregados: {len(trips)} filas → {_rel(outputs.trips_daily)}")
    telemetry = create_telemetry_summary(trips, outputs.telemetry_summary, outputs.trips_daily)
    print(f"[geotab] Resumen telemetría: {len(telemetry)} filas → {_rel(outputs.telemetry_summary)}")
    ingest_events(paths.logrecord, paths.status, devices, outputs.events_raw, outputs.events_daily, outputs.events_percentiles)
    save_rules_zones(paths.rules, paths.zones, outputs.rules, outputs.zones)
    print(f"[geotab] Rules/Zones exportados a {_rel(STAGING_DIR)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingesta de CSV Geotab → staging datasets")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="Directorio con Device.csv, FaultData.csv, etc.")
    parser.add_argument("--devices", type=Path, default=None, help="Ruta específica a Device.csv")
    parser.add_argument("--faults", type=Path, default=None, help="Ruta específica a FaultData.csv")
    parser.add_argument("--trips", type=Path, default=None, help="Ruta específica a Trip.csv")
    parser.add_argument("--rules", type=Path, default=None, help="Ruta específica a Rule.csv")
    parser.add_argument("--zones", type=Path, default=None, help="Ruta específica a Zone.csv")
    parser.add_argument("--users", type=Path, default=None, help="Ruta específica a User.csv (opcional)")
    parser.add_argument("--logrecord", type=Path, default=None, help="Ruta específica a LogRecord.csv (opcional)")
    parser.add_argument("--status", type=Path, default=None, help="Ruta específica a StatusData.csv (opcional)")
    parser.add_argument("--out-suffix", type=str, default=None, help="Sufijo para versionar salidas (ej: 2025-12-21)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.source.expanduser().resolve()
    if not source.exists():
        raise SystemExit(f"Directorio fuente no encontrado: {source}")
    suffix = args.out_suffix
    paths = Paths(
        devices=(args.devices or source / "Device.csv"),
        faults=(args.faults or source / "FaultData.csv"),
        trips=(args.trips or source / "Trip.csv"),
        rules=(args.rules or source / "Rule.csv"),
        zones=(args.zones or source / "Zone.csv"),
        users=(args.users or None),
        logrecord=(args.logrecord or source / "LogRecord.csv"),
        status=(args.status or source / "StatusData.csv"),
    )
    outputs = Outputs(
        devices=_with_suffix(GEOTAB_DEVICES_OUT, suffix),
        faults=_with_suffix(GEOTAB_FAULTS_OUT, suffix),
        trips_daily=_with_suffix(GEOTAB_TRIPS_DAILY_OUT, suffix),
        trips_raw=_with_suffix(GEOTAB_TRIPS_RAW_OUT, suffix),
        rules=_with_suffix(GEOTAB_RULES_OUT, suffix),
        zones=_with_suffix(GEOTAB_ZONES_OUT, suffix),
        telemetry_summary=_with_suffix(TELEMETRY_SUMMARY_OUT, suffix),
        events_raw=_with_suffix(GEOTAB_EVENTS_RAW_OUT, suffix),
        events_daily=_with_suffix(GEOTAB_EVENTS_DAILY_OUT, suffix),
        events_percentiles=_with_suffix(GEOTAB_EVENTS_MARKET_PCTL_OUT, suffix),
    )
    ingest(paths, outputs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
