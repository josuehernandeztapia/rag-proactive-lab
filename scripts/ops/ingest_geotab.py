#!/usr/bin/env python3
"""Normaliza los CSV exportados de Geotab y genera staging datasets compartidos.

Usage:
    python scripts/ops/ingest_geotab.py --source data/raw/geotab
    python scripts/ops/ingest_geotab.py \
        --devices data/raw/geotab/Device.csv \
        --faults data/raw/geotab/FaultData.csv \
        --trips data/raw/geotab/Trip.csv \
        --rules data/raw/geotab/Rule.csv \
        --zones data/raw/geotab/Zone.csv

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


@dataclass
class Paths:
    devices: Path
    faults: Path
    trips: Path
    rules: Path
    zones: Path
    users: Path | None = None


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


def _ensure_output_dirs() -> None:
    STAGING_DIR.mkdir(parents=True, exist_ok=True)


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def load_devices(path: Path) -> pd.DataFrame:
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
    subset.to_csv(GEOTAB_DEVICES_OUT, index=False)
    write_metadata(
        GEOTAB_DEVICES_OUT,
        script=__file__,
        inputs=[path],
        extra={'rows': int(len(subset))}
    )
    return subset


def normalize_faults(path: Path, devices: pd.DataFrame) -> pd.DataFrame:
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
    normalized.to_csv(GEOTAB_FAULTS_OUT, index=False)
    write_metadata(
        GEOTAB_FAULTS_OUT,
        script=__file__,
        inputs=[path, GEOTAB_DEVICES_OUT],
        extra={'rows': int(len(normalized))}
    )
    return normalized


def _to_seconds(series: pd.Series) -> pd.Series:
    return pd.to_timedelta(series, errors="coerce").dt.total_seconds()


def normalize_trips(path: Path, devices: pd.DataFrame) -> pd.DataFrame:
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
    grouped.to_csv(GEOTAB_TRIPS_DAILY_OUT, index=False)

    write_metadata(
        GEOTAB_TRIPS_DAILY_OUT,
        script=__file__,
        inputs=[path, GEOTAB_DEVICES_OUT],
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
    df[keep_raw_cols].to_csv(GEOTAB_TRIPS_RAW_OUT, index=False)
    write_metadata(
        GEOTAB_TRIPS_RAW_OUT,
        script=__file__,
        inputs=[path],
        extra={'rows': int(len(df))}
    )
    return grouped


def create_telemetry_summary(trip_daily: pd.DataFrame) -> pd.DataFrame:
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
    summary.to_csv(TELEMETRY_SUMMARY_OUT, index=False)
    write_metadata(
        TELEMETRY_SUMMARY_OUT,
        script=__file__,
        inputs=[GEOTAB_TRIPS_DAILY_OUT],
        extra={'rows': int(len(summary))}
    )
    return summary


def save_rules_zones(rules_path: Path, zones_path: Path) -> None:
    pd.read_csv(rules_path).to_csv(GEOTAB_RULES_OUT, index=False)
    pd.read_csv(zones_path).to_csv(GEOTAB_ZONES_OUT, index=False)
    write_metadata(GEOTAB_RULES_OUT, script=__file__, inputs=[rules_path])
    write_metadata(GEOTAB_ZONES_OUT, script=__file__, inputs=[zones_path])


def ingest(paths: Paths) -> None:
    _ensure_output_dirs()
    devices = load_devices(paths.devices)
    print(f"[geotab] Dispositivos normalizados → {_rel(GEOTAB_DEVICES_OUT)}")
    faults = normalize_faults(paths.faults, devices)
    print(f"[geotab] Faults normalizados: {len(faults)} filas → {_rel(GEOTAB_FAULTS_OUT)}")
    trips = normalize_trips(paths.trips, devices)
    print(f"[geotab] Trips agregados: {len(trips)} filas → {_rel(GEOTAB_TRIPS_DAILY_OUT)}")
    telemetry = create_telemetry_summary(trips)
    print(f"[geotab] Resumen telemetría: {len(telemetry)} filas → {_rel(TELEMETRY_SUMMARY_OUT)}")
    save_rules_zones(paths.rules, paths.zones)
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.source.expanduser().resolve()
    if not source.exists():
        raise SystemExit(f"Directorio fuente no encontrado: {source}")
    paths = Paths(
        devices=(args.devices or source / "Device.csv"),
        faults=(args.faults or source / "FaultData.csv"),
        trips=(args.trips or source / "Trip.csv"),
        rules=(args.rules or source / "Rule.csv"),
        zones=(args.zones or source / "Zone.csv"),
        users=(args.users or None),
    )
    ingest(paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
