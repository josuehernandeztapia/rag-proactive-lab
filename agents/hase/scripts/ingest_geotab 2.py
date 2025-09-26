#!/usr/bin/env python3
"""Normalize Geotab reports (Trips + Engine) into staging CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd

DEFAULT_TRIPS_COLUMNS = {
    "placa": "placa",
    "fecha_dia": "fecha_dia",
    "downtime_hours": "downtime_hours",
    "activity_drop_pct": "activity_drop_pct",
}

DEFAULT_ENGINE_COLUMNS = {
    "placa": "placa",
    "fecha_dia": "fecha_dia",
    "engine_events": "engine_events",
}


MONTH_MAP = {
    "ene.": "Jan",
    "feb.": "Feb",
    "mar.": "Mar",
    "abr.": "Apr",
    "may.": "May",
    "jun.": "Jun",
    "jul.": "Jul",
    "ago.": "Aug",
    "sep.": "Sep",
    "oct.": "Oct",
    "nov.": "Nov",
    "dic.": "Dec",
}


def parse_spanish_datetime(value: str) -> pd.Timestamp:
    if pd.isna(value) or isinstance(value, float):
        return pd.NaT
    parts = value.split(" ")
    if len(parts) >= 4 and parts[0].lower() in MONTH_MAP:
        parts[0] = MONTH_MAP[parts[0].lower()]
        normalized = " ".join(parts)
        return pd.to_datetime(normalized, errors="coerce", format="%b %d, %Y %I:%M:%S %p")
    return pd.to_datetime(value, errors="coerce")


def normalize_plate(value: str, mapping: dict[str, str]) -> str:
    if pd.isna(value):
        return ""
    value = str(value).strip()
    key = ''.join(ch for ch in value if ch.isalnum()).upper()
    return mapping.get(key, key)


def parse_duration(duration: str) -> float:
    if pd.isna(duration) or duration == "":
        return 0.0
    parts = duration.split(":")
    try:
        if len(parts) == 2:  # MM:SS
            h = 0
            m, s = parts
        elif len(parts) == 3:  # HH:MM:SS
            h, m, s = parts
        else:
            return 0.0
        return int(h) + int(m) / 60 + int(s) / 3600
    except ValueError:
        return 0.0


def load_mapping(path: Path | None) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if path and path.exists():
        df_map = pd.read_csv(path)
        for _, row in df_map.iterrows():
            raw = str(row[0]).strip()
            plate = str(row[1]).strip()
            mapping[''.join(ch for ch in raw if ch.isalnum()).upper()] = ''.join(ch for ch in plate if ch.isalnum()).upper()
    return mapping


def ingest_trips(path: Path, mapping: dict[str, str]) -> pd.DataFrame:
    df_raw = pd.read_csv(path, header=None)
    header_index = df_raw.index[df_raw.iloc[:, 0] == "Dispositivo"].tolist()
    if not header_index:
        raise ValueError("No se encontró cabecera 'Dispositivo' en el reporte de Trips")
    header_row = header_index[0]
    df = pd.read_csv(path, header=header_row)

    for col in ("Fecha de inicio", "Fecha de parada"):
        df[col] = df[col].apply(parse_spanish_datetime)

    df["downtime_hours"] = df["Duración de la parad"].apply(parse_duration)
    df["placa"] = df["Dispositivo"].apply(lambda val: normalize_plate(val, mapping))
    df["fecha_dia"] = df["Fecha de parada"].dt.date

    telemetry = (
        df.dropna(subset=["placa", "fecha_dia"])  # drop rows without placa/fecha
          .groupby(["placa", "fecha_dia"], as_index=False)
          .agg(
              downtime_hours=("downtime_hours", "sum"),
          )
    )
    telemetry["activity_drop_pct"] = pd.NA
    return telemetry.rename(columns=DEFAULT_TRIPS_COLUMNS)


def ingest_engine(path: Path, mapping: dict[str, str]) -> pd.DataFrame:
    df_raw = pd.read_csv(path, header=None)
    header_index = df_raw.index[df_raw.iloc[:, 0] == "Dispositivo"].tolist()
    if not header_index:
        raise ValueError("No se encontró cabecera 'Dispositivo' en el reporte de Engine")
    header_row = header_index[0]
    df = pd.read_csv(path, header=header_row)

    df["placa"] = df["Dispositivo"].apply(lambda val: normalize_plate(val, mapping))
    df["fecha_dia"] = df["Fecha"].apply(lambda val: parse_spanish_datetime(val).date() if pd.notnull(val) else pd.NaT)

    engine = (
        df.dropna(subset=["placa", "fecha_dia"])  # drop rows without placa/fecha
          .groupby(["placa", "fecha_dia"], as_index=False)
          .agg(engine_events=("Descripción", "count"))
    )
    return engine.rename(columns=DEFAULT_ENGINE_COLUMNS)


def write_output(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize Geotab reports")
    parser.add_argument("--trips", type=Path, required=True, help="CSV export from Advanced Trips Detail Report")
    parser.add_argument("--engine", type=Path, required=True, help="CSV export from Advanced Engine Status Report")
    parser.add_argument("--mapping", type=Path, required=False, help="CSV mapping: dispositivo,placa real")
    parser.add_argument("--out-telemetry", type=Path, default=Path("data/staging/telemetry_summary.csv"))
    parser.add_argument("--out-engine", type=Path, default=Path("data/staging/engine_summary.csv"))
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    mapping = load_mapping(args.mapping)

    telemetry_df = ingest_trips(args.trips, mapping)
    write_output(telemetry_df, args.out_telemetry)

    engine_df = ingest_engine(args.engine, mapping)
    write_output(engine_df, args.out_engine)

    print(f"Telemetry summary -> {args.out_telemetry}")
    print(f"Engine summary    -> {args.out_engine}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
