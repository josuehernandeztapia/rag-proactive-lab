#!/usr/bin/env python3
"""Summaries of low GNV consumption by plaza and municipality."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize temporal consumption patterns")
    parser.add_argument(
        "--daily-features",
        type=Path,
        default=Path("data/hase/consumos_features_daily.csv.gz"),
        help="Daily aggregated dataset",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("notebooks/hase/consumos_temporal_summary.md"),
        help="Markdown output path",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="Number of lowest months/days to report",
    )
    return parser.parse_args(argv)


def load_daily(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    needed = {"plaza_limpia", "placa", "fecha_dia", "litros_diarios"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    df["fecha_dia"] = pd.to_datetime(df["fecha_dia"], errors="coerce")
    df = df.dropna(subset=["fecha_dia", "placa"])
    df["mes"] = df["fecha_dia"].dt.to_period("M").dt.to_timestamp()
    df["dow"] = df["fecha_dia"].dt.day_name()
    df["dia_calendario"] = df["fecha_dia"].dt.day
    return df


def format_markdown(df: pd.DataFrame, float_columns: list[str] | None = None) -> str:
    if float_columns is None:
        float_columns = []
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        cells = []
        for col in headers:
            val = row[col]
            if col in float_columns and pd.notna(val):
                cells.append(f"{float(val):.2f}")
            else:
                cells.append(str(val))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_month_summary(df: pd.DataFrame, top: int) -> pd.DataFrame:
    grouped = (
        df.groupby(["plaza_limpia", "mes"], as_index=False)
        .agg(
            litros_totales=("litros_diarios", "sum"),
            placas_unicas=("placa", "nunique"),
        )
    )
    grouped["litros_promedio_por_placa"] = grouped["litros_totales"] / grouped["placas_unicas"].replace(0, pd.NA)
    grouped = grouped.sort_values(["plaza_limpia", "litros_totales"])
    return (
        grouped.groupby("plaza_limpia")
        .head(top)
        .sort_values(["plaza_limpia", "mes"])
        .reset_index(drop=True)
    )


def build_weekday_summary(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["plaza_limpia", "dow"], as_index=False)
        .agg(
            litros_promedio=("litros_diarios", "mean"),
            placas_promedio=("placa", "nunique"),
        )
    )
    grouped = grouped.sort_values(["plaza_limpia", "litros_promedio"])
    return grouped.groupby("plaza_limpia").head(3).reset_index(drop=True)


def build_calendar_day_summary(df: pd.DataFrame, top: int) -> pd.DataFrame:
    grouped = (
        df.groupby(["plaza_limpia", "dia_calendario"], as_index=False)["litros_diarios"].mean()
        .rename(columns={"litros_diarios": "litros_promedio"})
    )
    grouped = grouped.sort_values(["plaza_limpia", "litros_promedio"])
    return grouped.groupby("plaza_limpia").head(top).reset_index(drop=True)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    df = load_daily(args.daily_features)

    month_summary = build_month_summary(df, args.top)
    weekday_summary = build_weekday_summary(df)
    calendar_day_summary = build_calendar_day_summary(df, args.top)

    output = [
        "# Consumo mínimo por plaza (histórico)",
        "",
        "## Meses con menor consumo total",
        format_markdown(
            month_summary[["plaza_limpia", "mes", "litros_totales", "placas_unicas", "litros_promedio_por_placa"]],
            float_columns=["litros_totales", "litros_promedio_por_placa"],
        ),
        "",
        "## Días de la semana con menor consumo promedio",
        format_markdown(
            weekday_summary[["plaza_limpia", "dow", "litros_promedio", "placas_promedio"]],
            float_columns=["litros_promedio"],
        ),
        "",
        "## Días del mes (calendario) con menor consumo promedio",
        format_markdown(
            calendar_day_summary[["plaza_limpia", "dia_calendario", "litros_promedio"]],
            float_columns=["litros_promedio"],
        ),
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(output), encoding="utf-8")
    print(f"Summary written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
