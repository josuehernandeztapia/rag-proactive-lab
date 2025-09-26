#!/usr/bin/env python3
"""Build consumption-based features for HASE and PIA models."""

from __future__ import annotations

import argparse
import pathlib
import sys
import unicodedata
from typing import Iterable

import numpy as np
import pandas as pd


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate rolling consumption features")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Paths to raw CSV files with historical consumption records",
    )
    parser.add_argument(
        "--daily-output",
        type=pathlib.Path,
        default=pathlib.Path("data/hase/consumos_features_daily.parquet"),
        help="Path where the daily feature table will be written",
    )
    parser.add_argument(
        "--snapshot-output",
        type=pathlib.Path,
        default=pathlib.Path("data/hase/consumos_snapshot_latest.parquet"),
        help="Path for the latest snapshot per placa",
    )
    parser.add_argument(
        "--plaza-summary",
        type=pathlib.Path,
        default=pathlib.Path("data/hase/consumos_summary_by_plaza.csv"),
        help="Optional aggregated metrics per plaza",
    )
    parser.add_argument(
        "--target-payment",
        type=float,
        default=11000.0,
        help="Monto esperado a cubrir con GNV por unidad (MXN)",
    )
    parser.add_argument(
        "--max-price",
        type=float,
        default=24.0,
        help="Precio tope del litro utilizado para simular sobreprecio (MXN)",
    )
    parser.add_argument(
        "--base-price",
        type=float,
        default=10.5,
        help="Precio base estimado del GNV cuando no hay recaudo (MXN)",
    )
    parser.add_argument(
        "--telemetry-summary",
        type=pathlib.Path,
        default=None,
        help="CSV opcional con resumen de telemetría por placa y día",
    )
    parser.add_argument(
        "--protection-usage",
        type=pathlib.Path,
        default=None,
        help="CSV opcional con uso reciente de Protección por placa",
    )
    return parser.parse_args(argv)


def _strip_accents(value: str) -> str:
    if value is None:
        return ""
    text = str(value)
    try:
        text = text.encode("latin1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(c for c in normalized if not unicodedata.combining(c))


def _clean_plaza(plaza: str, estacion: str) -> str:
    plaza_norm = _strip_accents(str(plaza)).upper().strip()
    estacion_norm = _strip_accents(str(estacion)).upper().strip()

    if plaza_norm in {"ESTADO DE MEXICO", "ESTADO DE MEXICO "}:
        if "ECATEPEC" in estacion_norm:
            return "EDOMEX - ECATEPEC"
        if "TLANEPANTLA" in estacion_norm:
            return "EDOMEX - TLANEPANTLA"
        return "EDOMEX - OTRO"
    if plaza_norm == "AGUASCALIENTES":
        return "AGUASCALIENTES"
    return plaza_norm or "DESCONOCIDO"


def _load_sources(paths: Iterable[str]) -> pd.DataFrame:
    frames = []
    for path in paths:
        df = pd.read_csv(path)
        frames.append(df)
    if not frames:
        raise ValueError("No input files provided")
    df_concat = pd.concat(frames, ignore_index=True)
    return df_concat


def _load_telemetry_summary(path: pathlib.Path | None) -> pd.DataFrame | None:
    if not path:
        return None
    telemetry = pd.read_csv(path)
    if telemetry.empty:
        return None
    if "placa" not in telemetry.columns or "fecha_dia" not in telemetry.columns:
        raise ValueError("Telemetry summary must include 'placa' and 'fecha_dia' columns")

    telemetry["fecha_dia"] = pd.to_datetime(telemetry["fecha_dia"], errors="coerce").dt.floor("D")
    telemetry = telemetry.dropna(subset=["placa", "fecha_dia"])

    downtime_col = next((c for c in ("downtime_hours", "downtime_hrs", "hours_down") if c in telemetry.columns), None)
    if downtime_col is None and "downtime_days" in telemetry.columns:
        telemetry["downtime_hours"] = telemetry["downtime_days"] * 24
    elif downtime_col is not None:
        telemetry["downtime_hours"] = telemetry[downtime_col]
    else:
        telemetry["downtime_hours"] = np.nan

    activity_col = next((c for c in ("activity_drop_pct", "activity_drop_percent", "activity_drop") if c in telemetry.columns), None)
    if activity_col is not None:
        telemetry["activity_drop_pct"] = telemetry[activity_col]
    else:
        telemetry["activity_drop_pct"] = np.nan

    telemetry_grouped = (
        telemetry.groupby(["placa", "fecha_dia"], as_index=False)
        .agg(
            downtime_hours=("downtime_hours", "sum"),
            activity_drop_pct=("activity_drop_pct", "mean"),
        )
    )
    return telemetry_grouped


def _load_protection_usage(path: pathlib.Path | None) -> pd.DataFrame | None:
    if not path:
        return None
    usage = pd.read_csv(path)
    if usage.empty:
        return None
    if "placa" not in usage.columns:
        raise ValueError("Protection usage file must include 'placa' column")

    if "last_protection_at" in usage.columns:
        usage["last_protection_at"] = pd.to_datetime(usage["last_protection_at"], errors="coerce")
    else:
        usage["last_protection_at"] = pd.NaT

    if "protections_applied_last_12m" not in usage.columns:
        usage["protections_applied_last_12m"] = np.nan

    usage_sorted = usage.sort_values("last_protection_at")
    usage_grouped = (
        usage_sorted.groupby("placa", as_index=False)
        .agg(
            last_protection_at=("last_protection_at", "last"),
            protections_applied_last_12m=("protections_applied_last_12m", "max"),
        )
    )
    return usage_grouped


def build_features(
    df: pd.DataFrame,
    target_payment: float,
    max_price: float,
    base_price: float,
    telemetry: pd.DataFrame | None = None,
    protection_usage: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    expected_columns = {
        "fecha_venta",
        "estacion_servicio",
        "plaza",
        "placa",
        "litros",
        "venta_total_recaudo",
        "pvp",
    }
    missing = expected_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["fecha_venta"] = pd.to_datetime(df["fecha_venta"], errors="coerce")
    df = df.dropna(subset=["fecha_venta", "placa"])
    df["fecha_dia"] = df["fecha_venta"].dt.floor("D")
    df["plaza_limpia"] = [
        _clean_plaza(plaza, estacion)
        for plaza, estacion in zip(df["plaza"], df["estacion_servicio"], strict=True)
    ]

    precio_estimado = df["pvp"].where(df["pvp"] > 0, np.nan).fillna(base_price)
    extra_credito = np.clip(max_price - precio_estimado, 0, None) * df["litros"].fillna(0)
    df["extra_credito"] = extra_credito

    daily = (
        df.groupby(["plaza_limpia", "placa", "fecha_dia"], as_index=False)
        .agg(
            litros_diarios=("litros", "sum"),
            tickets_diarios=("litros", "count"),
            recaudo_diario=("venta_total_recaudo", "sum"),
            precio_promedio=("pvp", "mean"),
            credito_diario=("extra_credito", "sum"),
        )
        .sort_values(["placa", "fecha_dia"])
    )

    # Rolling features per placa
    windows = [7, 14, 30]
    for window in windows:
        roll = (
            daily.groupby("placa", group_keys=False)
            .rolling(window=window, min_periods=1, on="fecha_dia")
            .agg(
                {
                    "litros_diarios": "sum",
                    "tickets_diarios": "sum",
                    "recaudo_diario": "sum",
                    "credito_diario": "sum",
                }
            )
        )
        daily[f"litros_{window}d"] = roll["litros_diarios"].values
        daily[f"tickets_{window}d"] = roll["tickets_diarios"].values
        daily[f"recaudo_{window}d"] = roll["recaudo_diario"].values
        daily[f"credito_{window}d"] = roll["credito_diario"].values

    if target_payment > 0:
        daily["coverage_ratio_7d"] = daily["credito_7d"] / target_payment
        daily["coverage_ratio_14d"] = daily["credito_14d"] / target_payment
        daily["coverage_ratio_30d"] = daily["credito_30d"] / target_payment
    else:
        daily["coverage_ratio_7d"] = np.nan
        daily["coverage_ratio_14d"] = np.nan
        daily["coverage_ratio_30d"] = np.nan

    daily["lag_fecha"] = daily.groupby("placa")["fecha_dia"].shift(1)
    daily["recencia_dias"] = (
        (daily["fecha_dia"] - daily["lag_fecha"]).dt.days.fillna(0).astype(int)
    )
    daily = daily.drop(columns=["lag_fecha"])

    if telemetry is not None:
        daily = daily.merge(telemetry, on=["placa", "fecha_dia"], how="left")
    if "downtime_hours" not in daily.columns:
        daily["downtime_hours"] = np.nan
    if "activity_drop_pct" not in daily.columns:
        daily["activity_drop_pct"] = np.nan

    # Completa telemetría faltante con heurísticas determinísticas basadas en cobertura GNV.
    coverage_7 = daily["coverage_ratio_7d"].fillna(0)
    coverage_14 = daily["coverage_ratio_14d"].fillna(0)

    missing_downtime = daily["downtime_hours"].isna()
    if missing_downtime.any():
        shortage = (1 - coverage_14.clip(upper=1)).clip(lower=0)
        daily.loc[missing_downtime, "downtime_hours"] = (shortage[missing_downtime] * 12).round(2)

    missing_activity = daily["activity_drop_pct"].isna()
    if missing_activity.any():
        drop_proxy = (1 - coverage_7.clip(upper=1)).clip(lower=0)
        daily.loc[missing_activity, "activity_drop_pct"] = drop_proxy[missing_activity].round(3)

    daily["downtime_hours"] = daily["downtime_hours"].fillna(0)
    daily["activity_drop_pct"] = daily["activity_drop_pct"].fillna(0)

    for window in windows:
        roll_downtime = (
            daily.groupby("placa", group_keys=False)
            .rolling(window=window, min_periods=1, on="fecha_dia")
            .agg({"downtime_hours": "sum"})
        )
        daily[f"downtime_hours_{window}d"] = roll_downtime["downtime_hours"].values
        daily[f"downtime_days_{window}d"] = daily[f"downtime_hours_{window}d"] / 24.0

    if protection_usage is not None:
        daily = daily.merge(protection_usage, on="placa", how="left")
    if "protections_applied_last_12m" not in daily.columns:
        daily["protections_applied_last_12m"] = np.nan
    if "last_protection_at" not in daily.columns:
        daily["last_protection_at"] = pd.NaT

    # Snapshot with the latest available day per placa
    latest_idx = daily.groupby("placa")["fecha_dia"].transform("max") == daily["fecha_dia"]
    snapshot = daily.loc[latest_idx].copy()

    plaza_summary = (
        daily.groupby("plaza_limpia", as_index=False)
        .agg(
            primera_fecha=("fecha_dia", "min"),
            ultima_fecha=("fecha_dia", "max"),
            registros=("fecha_dia", "count"),
            placas_unicas=("placa", "nunique"),
            litros_total=("litros_diarios", "sum"),
            recaudo_total=("recaudo_diario", "sum"),
            cobertura_media_14d=("coverage_ratio_14d", "mean"),
            cobertura_media_30d=("coverage_ratio_30d", "mean"),
        )
        .sort_values("plaza_limpia")
    )

    return daily, snapshot, plaza_summary


def _safe_write_frame(df: pd.DataFrame, path: pathlib.Path) -> None:
    """Write dataframe handling parquet fallback when pyarrow is unavailable."""
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        try:
            df.to_parquet(path, index=False)
            return
        except ImportError:
            alt_path = path.with_suffix(".csv.gz")
            df.to_csv(alt_path, index=False, compression="gzip")
            print(
                f"WARN: pyarrow not available, wrote CSV.GZ fallback instead -> {alt_path}",
                file=sys.stderr,
            )
            return
    if suffix == ".gz":
        df.to_csv(path, index=False, compression="gzip")
    else:
        df.to_csv(path, index=False)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    raw_df = _load_sources(args.inputs)
    telemetry_df = _load_telemetry_summary(args.telemetry_summary)
    protection_usage_df = _load_protection_usage(args.protection_usage)
    daily, snapshot, plaza_summary = build_features(
        raw_df,
        args.target_payment,
        args.max_price,
        args.base_price,
        telemetry_df,
        protection_usage_df,
    )

    args.daily_output.parent.mkdir(parents=True, exist_ok=True)
    args.snapshot_output.parent.mkdir(parents=True, exist_ok=True)
    args.plaza_summary.parent.mkdir(parents=True, exist_ok=True)

    _safe_write_frame(daily, args.daily_output)
    _safe_write_frame(snapshot, args.snapshot_output)
    plaza_summary.to_csv(args.plaza_summary, index=False)

    print(
        "Daily features saved to "
        f"{args.daily_output.with_suffix('.csv.gz') if args.daily_output.suffix == '.parquet' else args.daily_output}"
    )
    print(
        "Snapshot saved to "
        f"{args.snapshot_output.with_suffix('.csv.gz') if args.snapshot_output.suffix == '.parquet' else args.snapshot_output}"
    )
    print(f"Plaza summary saved to {args.plaza_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
