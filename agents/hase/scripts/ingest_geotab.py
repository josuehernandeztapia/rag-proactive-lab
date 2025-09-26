#!/usr/bin/env python3
"""Normalize Geotab API CSV exports into staging files."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd


SPANISH_MONTHS = {
    'ene.': 'Jan', 'feb.': 'Feb', 'mar.': 'Mar', 'abr.': 'Apr', 'may.': 'May', 'jun.': 'Jun',
    'jul.': 'Jul', 'ago.': 'Aug', 'sep.': 'Sep', 'oct.': 'Oct', 'nov.': 'Nov', 'dic.': 'Dec'
}


def parse_spanish_datetime(value: str) -> pd.Timestamp:
    if pd.isna(value) or value == '':
        return pd.NaT
    for short, eng in SPANISH_MONTHS.items():
        if value.lower().startswith(short):
            value = value.replace(short, eng, 1)
            break
    return pd.to_datetime(value, errors='coerce', format='%b %d, %Y %I:%M:%S %p')


def normalize_device_id(value) -> str | None:
    if isinstance(value, str) and value.startswith("{'id':"):
        try:
            return eval(value)['id']
        except Exception:
            return None
    return value if isinstance(value, str) else None


def map_device_to_plate(device_path: Path) -> dict[str, str]:
    df = pd.read_csv(device_path)
    mapping = {}
    for _, row in df.iterrows():
        device_id = row.get('id')
        plate = row.get('licensePlate') or row.get('name')
        if isinstance(device_id, str):
            mapping[device_id] = plate
    return mapping


def ingest_trips(trip_path: Path, device_to_plate: dict[str, str]) -> pd.DataFrame:
    df = pd.read_csv(trip_path)
    df['device_id'] = df['device'].apply(normalize_device_id)
    df['placa'] = df['device_id'].map(device_to_plate)

    if 'stop' in df:
        df['fecha_dia'] = pd.to_datetime(df['stop'], errors='coerce').dt.date
    else:
        df['fecha_dia'] = pd.NaT

    df['distance'] = df['distance'].fillna(0)

    def to_hours(series: pd.Series) -> float:
        if pd.isna(series):
            return 0.0
        try:
            td = pd.to_timedelta(series)
            return td.total_seconds() / 3600.0
        except ValueError:
            if isinstance(series, str) and series.count('.') == 1 and ':' in series:
                day_part, time_part = series.split('.', 1)
                try:
                    days = int(day_part)
                except ValueError:
                    days = 0
                time_td = pd.to_timedelta(time_part)
                return days * 24 + time_td.total_seconds() / 3600.0
            return 0.0

    df['work_stop_hours'] = df['workStopDuration'].apply(to_hours)
    df['work_drive_hours'] = df['workDrivingDuration'].apply(to_hours)
    df['idling_hours'] = df['idlingDuration'].apply(to_hours)

    grouped = (
        df.dropna(subset=['placa', 'fecha_dia'])
          .groupby(['placa', 'fecha_dia'], as_index=False)
          .agg(
              viajes=('distance', 'size'),
              distancia_total=('distance', 'sum'),
              horas_trabajadas=('work_drive_hours', 'sum'),
              horas_paradas=('work_stop_hours', 'sum'),
              horas_idling=('idling_hours', 'sum'),
          )
          .sort_values(['placa', 'fecha_dia'])
    )

    # Interpret downtime as la suma de horas paradas + idling en cada día.
    grouped['downtime_hours'] = grouped['horas_paradas'].fillna(0) + grouped['horas_idling'].fillna(0)

    # Aproximación de caída de actividad vs. promedio móvil de 14 días.
    rolling_mean = (
        grouped.groupby('placa', group_keys=False)['horas_trabajadas']
        .transform(lambda s: s.rolling(window=14, min_periods=1).mean())
    )
    ratio = grouped['horas_trabajadas'] / rolling_mean.replace(0, pd.NA)
    grouped['activity_drop_pct'] = (1 - ratio).clip(lower=0).fillna(0)

    return grouped


def ingest_faults(fault_path: Path, device_to_plate: dict[str, str]) -> pd.DataFrame:
    df = pd.read_csv(fault_path)
    df['device_id'] = df['device'].apply(normalize_device_id)
    df['placa'] = df['device_id'].map(device_to_plate)
    df['fecha_dia'] = pd.to_datetime(df['dateTime'], errors='coerce').dt.date

    grouped = (
        df.dropna(subset=['placa', 'fecha_dia'])
          .groupby(['placa', 'fecha_dia'], as_index=False)
          .agg(
              engine_events=('diagnostic', 'size'),
              fault_active=('faultState', lambda s: (s == 'Active').sum()),
          )
    )
    return grouped


def write_output(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize Geotab API exports")
    parser.add_argument("--device", type=Path, required=True)
    parser.add_argument("--trip", type=Path, required=True)
    parser.add_argument("--fault", type=Path, required=True)
    parser.add_argument("--out-telemetry", type=Path, default=Path('data/staging/telemetry_summary.csv'))
    parser.add_argument("--out-engine", type=Path, default=Path('data/staging/engine_summary.csv'))
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    device_to_plate = map_device_to_plate(args.device)
    telemetry_df = ingest_trips(args.trip, device_to_plate)
    write_output(telemetry_df, args.out_telemetry)

    engine_df = ingest_faults(args.fault, device_to_plate)
    write_output(engine_df, args.out_engine)

    print(f"Telemetry summary -> {args.out_telemetry}")
    print(f"Engine summary    -> {args.out_engine}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
