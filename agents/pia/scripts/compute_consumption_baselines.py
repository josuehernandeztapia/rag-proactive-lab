#!/usr/bin/env python3
"""Construir baselines de consumo GNV por plaza y placa."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE_DIR = ROOT.parent / "1.-Consumos GNV Aguascalientes y Edomex"
DEFAULT_OUTPUT = ROOT / "data" / "pia" / "hase_consumption_baselines.csv"


@dataclass(frozen=True)
class SourceConfig:
    name: str
    path: Path
    plaza: str


NUMERIC_COLS = {"litros", "valor_recaudo", "recaudo_pagado", "total_precio_gnv"}
DATE_COL = "fecha_venta"
ID_COL = "placa"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generar baselines de consumo GNV por placa")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help="Directorio base con los archivos consolidados (AGS/EDOMEX)",
    )
    parser.add_argument(
        "--ags",
        type=Path,
        default=None,
        help="Archivo consolidado de AGS (CSV o CSV.GZ). Por defecto busca en source-dir.",
    )
    parser.add_argument(
        "--edomex",
        type=Path,
        default=None,
        help="Archivo consolidado de EDOMEX (CSV o CSV.GZ). Por defecto busca en source-dir.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Ruta de salida para el CSV con baselines.",
    )
    return parser.parse_args()


def resolve_sources(args: argparse.Namespace) -> list[SourceConfig]:
    sources: list[SourceConfig] = []
    if args.ags is not None:
        ags_path = args.ags
    else:
        ags_path = args.source_dir / "consumos_consolidados_unificado.ags.csv.gz"
        if not ags_path.exists():
            ags_path = args.source_dir / "consumos_consolidados_unificado.csv"
    if ags_path.exists():
        sources.append(SourceConfig(name="AGS", path=ags_path, plaza="AGUASCALIENTES"))

    if args.edomex is not None:
        edo_path = args.edomex
    else:
        edo_path = args.source_dir / "consumos_consolidados_unificado.edomex.csv.gz"
        if not edo_path.exists():
            edo_path = args.source_dir / "consumos_consolidados_unificado.edomex.csv"
    if edo_path.exists():
        sources.append(SourceConfig(name="EDOMEX", path=edo_path, plaza="EDOMEX"))

    if not sources:
        raise FileNotFoundError("No se encontraron archivos consolidados de consumo GNV")
    return sources


def load_dataset(config: SourceConfig) -> pd.DataFrame:
    df = pd.read_csv(config.path, encoding="utf-8", compression="infer")
    if DATE_COL not in df.columns:
        raise ValueError(f"El archivo {config.path} no contiene la columna {DATE_COL}")
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL])
    if ID_COL not in df.columns:
        raise ValueError(f"El archivo {config.path} no contiene la columna {ID_COL}")
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        else:
            df[col] = 0.0
    df["plaza"] = df.get("plaza", config.plaza).fillna(config.plaza)
    return df


def summarise_group(group: pd.DataFrame) -> dict[str, float | str]:
    group = group.sort_values(DATE_COL)
    first_date = group[DATE_COL].min()
    last_date = group[DATE_COL].max()
    res: dict[str, float | str] = {
        "placa": group[ID_COL].iloc[0],
        "plaza": group["plaza"].mode().iat[0] if not group["plaza"].mode().empty else "",
        "first_record_date": first_date.date().isoformat(),
        "last_record_date": last_date.date().isoformat(),
        "records_total": int(len(group)),
        "litros_total": float(group["litros"].sum()),
        "recaudo_total": float(group["valor_recaudo"].sum()),
    }

    time_span_days = max((last_date - first_date).days, 1)
    res["avg_daily_litros"] = res["litros_total"] / time_span_days
    res["avg_daily_recaudo"] = res["recaudo_total"] / time_span_days

    # ventanas móviles
    for window in (30, 90, 180, 365):
        mask = group[DATE_COL] >= (last_date - pd.Timedelta(days=window))
        window_df = group.loc[mask]
        res[f"litros_{window}d"] = float(window_df["litros"].sum())
        res[f"recaudo_{window}d"] = float(window_df["valor_recaudo"].sum())
        res[f"records_{window}d"] = int(len(window_df))

    # agregados por periodos de 30 días (resample)
    period = group.set_index(DATE_COL).resample("30D").sum()[["litros", "valor_recaudo"]]
    if not period.empty:
        res["avg_30d_litros"] = float(period["litros"].mean())
        res["avg_30d_recaudo"] = float(period["valor_recaudo"].mean())
        res["max_30d_litros"] = float(period["litros"].max())
        res["max_30d_recaudo"] = float(period["valor_recaudo"].max())
    else:
        res["avg_30d_litros"] = 0.0
        res["avg_30d_recaudo"] = 0.0
        res["max_30d_litros"] = 0.0
        res["max_30d_recaudo"] = 0.0

    return res


def build_baselines(datasets: Iterable[pd.DataFrame]) -> pd.DataFrame:
    merged = pd.concat(datasets, ignore_index=True)
    grouped = merged.groupby(ID_COL, dropna=False)
    summaries = [summarise_group(group) for _, group in grouped]
    return pd.DataFrame(summaries)


def main() -> int:
    args = parse_args()
    sources = resolve_sources(args)
    datasets = [load_dataset(cfg) for cfg in sources]
    baselines = build_baselines(datasets)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    baselines.to_csv(args.output, index=False)
    print(f"Baselines guardados en {args.output} ({len(baselines)} placas)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
