#!/usr/bin/env python3
"""Analyze GNV consumption vs. simulated credit coverage and highlight drops."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

DEFAULT_MAX_PRICE = 24.0  # MXN/litro (tope equivalente gasolina)
DEFAULT_TARGET_PAYMENT = 27000  # MXN mensuales a cubrir vía sobreprecio
ROLLING_WINDOW_MONTHS = 3
BASE_PRICE_FALLBACK = 10.5  # precio estimado cuando no hay recaudo (MXN)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate credit coverage using GNV consumption")
    parser.add_argument(
        "--daily-features",
        type=Path,
        default=Path("data/hase/consumos_features_daily.csv.gz"),
        help="Daily aggregated features (output of build_consumo_features.py)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("notebooks/hase/consumos_credito_analysis.md"),
        help="Markdown report destination",
    )
    parser.add_argument(
        "--target-payment",
        type=float,
        default=DEFAULT_TARGET_PAYMENT,
        help="Expected mensualidad a cubrir vía consumo GNV",
    )
    parser.add_argument(
        "--max-price",
        type=float,
        default=DEFAULT_MAX_PRICE,
        help="Precio máximo por litro (tope gasolina)",
    )
    parser.add_argument(
        "--base-price-fallback",
        type=float,
        default=BASE_PRICE_FALLBACK,
        help="Precio base a usar cuando no hay datos de recaudo",
    )
    return parser.parse_args(argv)


def load_daily(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Daily features not found: {path}")
    df = pd.read_csv(path)
    required = {"plaza_limpia", "placa", "fecha_dia", "litros_diarios", "recaudo_diario", "precio_promedio"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Daily dataset missing columns: {sorted(missing)}")
    df["fecha_dia"] = pd.to_datetime(df["fecha_dia"], errors="coerce")
    df = df.dropna(subset=["fecha_dia", "placa"]).copy()
    return df


def compute_credit_metrics(
    df: pd.DataFrame, max_price: float, base_price_fallback: float, target_payment: float
) -> pd.DataFrame:
    df = df.copy()
    df["litros_diarios"] = df["litros_diarios"].fillna(0)
    df["recaudo_diario"] = df["recaudo_diario"].fillna(0)

    precio_base = df["precio_promedio"].where(df["precio_promedio"] > 0, np.nan)
    precio_recaudo = np.divide(
        df["recaudo_diario"],
        df["litros_diarios"],
        out=np.zeros(len(df)),
        where=df["litros_diarios"] > 0,
    )
    precio_recaudo = np.where(precio_recaudo > 0, precio_recaudo, np.nan)
    precio_est = pd.Series(precio_base).fillna(pd.Series(precio_recaudo)).fillna(base_price_fallback)

    precio_est = np.clip(precio_est, 0, max_price)
    extra_por_litro = np.maximum(0.0, max_price - precio_est)
    df["credito_diario_gnv"] = df["litros_diarios"] * extra_por_litro

    df["mes"] = df["fecha_dia"].dt.to_period("M").dt.to_timestamp()

    monthly = (
        df.groupby(["plaza_limpia", "placa", "mes"], as_index=False)
        .agg(
            litros_mes=("litros_diarios", "sum"),
            recaudo_mes=("recaudo_diario", "sum"),
            credito_mes=("credito_diario_gnv", "sum"),
            dias_activos=("fecha_dia", "count"),
        )
        .sort_values(["placa", "mes"])
    )

    monthly["coverage_ratio"] = monthly["credito_mes"] / target_payment
    monthly["coverage_ratio"] = monthly["coverage_ratio"].fillna(0)
    monthly["shortfall"] = target_payment - monthly["credito_mes"]
    monthly["shortfall"] = monthly["shortfall"].clip(lower=0)

    monthly["coverage_rolling"] = (
        monthly.groupby("placa")["coverage_ratio"]
        .transform(lambda s: s.rolling(ROLLING_WINDOW_MONTHS, min_periods=1).mean())
    )
    monthly["coverage_change"] = monthly.groupby("placa")["coverage_ratio"].diff()

    return monthly


def summarize(monthly: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    month_summary = (
        monthly.groupby("mes", as_index=False)
        .agg(
            avg_coverage=("coverage_ratio", "mean"),
            median_coverage=("coverage_ratio", "median"),
            plates=("placa", "nunique"),
            pct_below_80=("coverage_ratio", lambda s: float((s < 0.8).mean() * 100)),
        )
        .sort_values("mes")
    )
    month_summary["delta_vs_prev"] = month_summary["avg_coverage"].diff()

    plaza_month_summary = (
        monthly.groupby(["plaza_limpia", "mes"], as_index=False)
        .agg(
            avg_coverage=("coverage_ratio", "mean"),
            median_coverage=("coverage_ratio", "median"),
            plates=("placa", "nunique"),
            pct_below_80=("coverage_ratio", lambda s: float((s < 0.8).mean() * 100)),
        )
        .sort_values(["plaza_limpia", "mes"])
    )
    plaza_month_summary["delta_vs_prev"] = plaza_month_summary.groupby("plaza_limpia")["avg_coverage"].diff()

    top_plate_drops = (
        monthly.groupby(["plaza_limpia", "placa"], as_index=False)
        .agg(
            worst_ratio=("coverage_ratio", "min"),
            worst_shortfall=("shortfall", "max"),
            last_ratio=("coverage_ratio", "last"),
            meses_registrados=("mes", "count"),
        )
        .sort_values("worst_ratio")
        .head(20)
    )

    return month_summary, plaza_month_summary, top_plate_drops


def _format_float(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def df_to_markdown(df: pd.DataFrame, float_cols: set[str] | None = None, digits: int = 2) -> str:
    if float_cols is None:
        float_cols = set(df.select_dtypes(include=["float", "float64", "float32"]).columns)
    headers = list(df.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in df.iterrows():
        cells = []
        for col in headers:
            val = row[col]
            if col in float_cols and isinstance(val, (int, float, np.floating)):
                cells.append(_format_float(float(val), digits))
            else:
                cells.append(str(val))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(
    output_path: Path,
    monthly: pd.DataFrame,
    month_summary: pd.DataFrame,
    plaza_month_summary: pd.DataFrame,
    top_plate_drops: pd.DataFrame,
    target_payment: float,
    max_price: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    overall_mean = monthly["coverage_ratio"].mean()
    overall_median = monthly["coverage_ratio"].median()
    months_below_80 = int((month_summary["avg_coverage"] < 0.8).sum())

    worst_months = (
        month_summary.sort_values("avg_coverage").head(10)[
            ["mes", "avg_coverage", "pct_below_80", "plates"]
        ]
    )
    biggest_drops = (
        month_summary.sort_values("delta_vs_prev").head(10)[
            ["mes", "avg_coverage", "delta_vs_prev", "pct_below_80", "plates"]
        ]
    )

    plaza_worst = (
        plaza_month_summary.sort_values("avg_coverage").groupby("plaza_limpia").head(5)[
            ["plaza_limpia", "mes", "avg_coverage", "pct_below_80", "plates"]
        ]
    )

    md_lines = [
        "# Consumo GNV vs Cobertura de Crédito",
        "",
        f"- Target mensual simulado: **${target_payment:,.0f} MXN**",
        f"- Precio tope gasolina aplicado: **${max_price:.2f} MXN/L**",
        f"- Cobertura media (razón extra/target): **{overall_mean:.2f}**",
        f"- Cobertura mediana: **{overall_median:.2f}**",
        f"- Meses con cobertura promedio < 0.80: **{months_below_80}**",
        "",
        "## Peores meses (cobertura promedio)",
        df_to_markdown(worst_months, float_cols={"avg_coverage", "pct_below_80"}),
        "",
        "## Mayores caídas mes vs mes",
        df_to_markdown(biggest_drops, float_cols={"avg_coverage", "delta_vs_prev", "pct_below_80"}),
        "",
        "## Peores meses por plaza",
        df_to_markdown(plaza_worst, float_cols={"avg_coverage", "pct_below_80"}),
        "",
        "## Placas con menor cobertura histórica",
        df_to_markdown(
            top_plate_drops[["plaza_limpia", "placa", "worst_ratio", "last_ratio", "worst_shortfall", "meses_registrados"]],
            float_cols={"worst_ratio", "last_ratio", "worst_shortfall"},
        ),
    ]

    output_path.write_text("\n".join(md_lines), encoding="utf-8")


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    daily = load_daily(args.daily_features)
    monthly = compute_credit_metrics(daily, args.max_price, args.base_price_fallback, args.target_payment)
    month_summary, plaza_month_summary, top_plate_drops = summarize(monthly)
    write_report(
        args.output,
        monthly,
        month_summary,
        plaza_month_summary,
        top_plate_drops,
        args.target_payment,
        args.max_price,
    )
    print(f"Report written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
