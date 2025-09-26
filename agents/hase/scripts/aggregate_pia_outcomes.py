#!/usr/bin/env python3
"""Genera features agregados de outcomes PIA para alimentar HASE."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from agents.pia.src.outcomes import DEFAULT_LOG_PATH, aggregate_outcomes, load_outcomes

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = ROOT / "data" / "hase" / "pia_outcomes_features.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Agregar outcomes de PIA por placa")
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG_PATH, help="Ruta del log de outcomes PIA")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Ruta de salida para los features")
    parser.add_argument(
        "--windows",
        type=int,
        nargs="*",
        default=[30, 90, 180],
        help="Ventanas móviles en días para el recuento de outcomes",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=None,
        help="Ruta opcional para guardar un resumen por plan de protección",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    df = load_outcomes(args.log)
    aggregated = aggregate_outcomes(df, windows=args.windows)
    if "protections_remaining" not in aggregated.columns:
        aggregated["protections_remaining"] = 0
    if "last_plan_status" not in aggregated.columns:
        aggregated["last_plan_status"] = ""
    if "last_plan_requires_manual_review" not in aggregated.columns:
        aggregated["last_plan_requires_manual_review"] = False
    aggregated["protections_flag_negative"] = aggregated["protections_remaining"].fillna(0) < 0
    aggregated["protections_flag_expired"] = aggregated["last_plan_status"].fillna("").str.lower().eq("expired")
    aggregated["protections_flag_manual"] = aggregated["last_plan_requires_manual_review"].fillna(False).astype(bool)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    aggregated.to_csv(args.output, index=False)
    unique_plans = (
        aggregated["last_plan_type"].dropna().unique().tolist()
        if "last_plan_type" in aggregated.columns
        else []
    )
    summary_msg = f"Features de outcomes guardados en {args.output} ({len(aggregated)} placas, planes: {', '.join(sorted(map(str, unique_plans))) or 'sin dato'})"
    negatives = aggregated["protections_flag_negative"].sum()
    expired = aggregated["protections_flag_expired"].sum()
    manual = aggregated["protections_flag_manual"].sum()
    if negatives or expired or manual:
        print(f"Alertas: negativos={negatives}, expirados={expired}, manual={manual}")
    print(summary_msg)

    if args.summary_out is not None:
        if "last_plan_type" not in aggregated.columns:
            aggregated["last_plan_type"] = "unknown"
        plan_summary = (
            aggregated.groupby("last_plan_type")
            .agg(
                contratos=("placa", "nunique"),
                protecciones_restantes_promedio=("protections_remaining", "mean"),
                protecciones_restantes_mediana=("protections_remaining", "median"),
                contratos_manual=("protections_flag_manual", "sum"),
                contratos_expirados=("protections_flag_expired", "sum"),
                contratos_negative=("protections_flag_negative", "sum"),
            )
            .reset_index()
            .rename(columns={"last_plan_type": "plan_type"})
        )
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        plan_summary.to_csv(args.summary_out, index=False)
        print(f"Resumen por plan guardado en {args.summary_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
