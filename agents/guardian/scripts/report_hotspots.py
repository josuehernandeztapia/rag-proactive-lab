#!/usr/bin/env python3
"""Resumen rápido de hotspots Guardian (seguridad, telemetría, etc.).

Usage:
    python agents/guardian/scripts/report_hotspots.py [--alerts safety telemetry] [--top 10]

Genera tablas resumidas por plaza y listado de placas con más incidencias
para facilitar la revisión humana.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.loader import get_config, get_path

DEFAULT_INSIGHTS_PATH = get_path('data', 'processed', 'guardian', 'insights')
DEFAULT_PIA_DATASET = get_path('data', 'processed', 'pia', 'features')


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Resumen de alertas Guardian por plaza/placa')
    parser.add_argument('--insights', type=Path, default=DEFAULT_INSIGHTS_PATH, help='CSV con guardian_insights')
    parser.add_argument('--pia-features', type=Path, default=DEFAULT_PIA_DATASET, help='Dataset PIA para mapear plazas')
    parser.add_argument('--alerts', nargs='*', default=None, help='Tipos de alerta a incluir (safety, telemetry, etc.)')
    parser.add_argument('--top', type=int, default=15, help='Número de placas top por volumen')
    parser.add_argument('--plaza-limit', type=int, default=20, help='Máximo de filas para la tabla por plaza')
    return parser.parse_args(argv)


def load_insights(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f'No se encontró el archivo: {_rel(path)}')
    df = pd.read_csv(path)
    if 'alert_type' not in df.columns:
        raise ValueError('guardian_insights.csv debe contener la columna alert_type')
    return df


def summarize_by_plaza(df: pd.DataFrame, limit: int) -> pd.DataFrame:
    plaza_summary = (
        df.groupby(['alert_type', 'placa_plaza'])
        .size()
        .reset_index(name='alertas')
        .sort_values(['alert_type', 'alertas'], ascending=[True, False])
    )
    if 'placa_plaza' not in df.columns:
        # intentar derivar plaza desde detalles cuando no exista
        plaza_summary = (
            df.assign(placa_plaza=df.get('plaza', 'UNKNOWN'))
            .groupby(['alert_type', 'placa_plaza'])
            .size()
            .reset_index(name='alertas')
            .sort_values(['alert_type', 'alertas'], ascending=[True, False])
        )
    return plaza_summary.groupby('alert_type').head(limit)


def summarize_by_plate(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    return (
        df.groupby(['alert_type', 'placa'])
        .size()
        .reset_index(name='alertas')
        .sort_values(['alert_type', 'alertas'], ascending=[True, False])
        .groupby('alert_type')
        .head(top_n)
    )


def main() -> int:
    args = parse_args()
    df = load_insights(args.insights)
    if args.pia_features.exists():
        pia_df = pd.read_csv(args.pia_features, usecols=['placa', 'plaza_limpia'])
        pia_df = pia_df.dropna(subset=['placa', 'plaza_limpia'])
        plaza_map = pia_df.drop_duplicates('placa').set_index('placa')['plaza_limpia']
        df['placa_plaza'] = df.get('placa').map(plaza_map).fillna(df.get('plaza', 'UNKNOWN'))
    else:
        df['placa_plaza'] = df.get('plaza', 'UNKNOWN')

    if args.alerts:
        mask = df['alert_type'].isin(args.alerts)
        df = df[mask]
        if df.empty:
            print('No hay alertas que coincidan con los filtros.')
            return 0
    else:
        # ordenar por severidad aproximada
        priority = pd.Categorical(
            df['alert_type'],
            categories=['safety', 'telemetry', 'idle', 'consumption', 'downtime', 'dtc'],
            ordered=True,
        )
        df = df.assign(alert_priority=priority)
        df = df.sort_values(['alert_priority', 'triggered_at']).drop(columns='alert_priority')

    # Derivar plaza cuando venga en detalles (para CSV legados)
    if 'placa_plaza' not in df.columns or df['placa_plaza'].isna().all():
        guardian_cfg = get_config('guardian', default={}) or {}
        df['placa_plaza'] = df.get('plaza', '').fillna('UNKNOWN')
        if 'plaza_map' in guardian_cfg:
            plaza_map_cfg = guardian_cfg['plaza_map'] or {}
            df['placa_plaza'] = df['placa_plaza'].replace(plaza_map_cfg)

    plaza_summary = summarize_by_plaza(df, args.plaza_limit)
    plate_summary = summarize_by_plate(df, args.top)

    print(f"Resumen Guardian → {_rel(args.insights)} (filtradas: {len(df)})")
    print('\n=== Alertas por plaza (Top) ===')
    if plaza_summary.empty:
        print('Sin alertas con plaza identificada.')
    else:
        print(plaza_summary.to_string(index=False))

    print(f"\n=== Placas con más alertas (Top {args.top}) ===")
    if plate_summary.empty:
        print('Sin alertas registradas con placa.')
    else:
        print(plate_summary.to_string(index=False))

    print("\n=== Últimas alertas ===")
    latest = df.sort_values('triggered_at', ascending=False).head(min(20, len(df)))
    cols = ['triggered_at', 'alert_type', 'placa', 'details', 'severity']
    cols = [c for c in cols if c in latest.columns]
    print(latest[cols].to_string(index=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
