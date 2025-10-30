#!/usr/bin/env python3
"""Genera un feed curado con hotspots de seguridad y telemetría para el dashboard."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.loader import get_path
from config.metadata import write_metadata

DEFAULT_DATASET = get_path('data', 'processed', 'pia', 'features_augmented')
DEFAULT_OUTPUT = get_path('data', 'processed', 'pia', 'hotspots')


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Construye el feed curado de hotspots PIA')
    parser.add_argument('--dataset', type=Path, default=DEFAULT_DATASET, help='Dataset PIA (features_augmented)')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT, help='Archivo CSV de salida')
    parser.add_argument('--top-safety', type=int, default=50, help='Placas con mayor riesgo de seguridad')
    parser.add_argument('--top-telemetry', type=int, default=50, help='Placas con telemetría crítica')
    return parser.parse_args(argv)


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f'Dataset no encontrado: {_rel(path)}')
    df = pd.read_csv(path)
    required = {'placa', 'plaza_limpia', 'safety_alert', 'telemetry_alert', 'seatbelt_off_rate_30d', 'high_speed_ratio_30d', 'telemetry_health_score'}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f'Dataset PIA carece de columnas requeridas: {sorted(missing)}')
    return df


def build_hotspots(df: pd.DataFrame, top_safety: int, top_telemetry: int) -> pd.DataFrame:
    dfs: list[pd.DataFrame] = []

    safety_df = df[df['safety_alert'] == 1].copy()
    if not safety_df.empty:
        safety_df = safety_df.sort_values(
            ['seatbelt_off_rate_30d', 'high_speed_ratio_30d'], ascending=[False, False]
        ).head(top_safety)
        safety_df['alert_type'] = 'safety'
        dfs.append(safety_df)

    telemetry_df = df[df['telemetry_alert'] == 1].copy()
    if not telemetry_df.empty:
        telemetry_df = telemetry_df.sort_values('telemetry_health_score', ascending=True).head(top_telemetry)
        telemetry_df['alert_type'] = 'telemetry'
        dfs.append(telemetry_df)

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    columns = [
        'alert_type',
        'placa',
        'plaza_limpia',
        'seatbelt_off_rate_30d',
        'high_speed_ratio_30d',
        'idle_hours_ratio_30d',
        'telemetry_health_score',
        'suggested_scenario',
        'risk_score',
        'needs_protection',
    ]
    available = [col for col in columns if col in combined.columns]
    return combined[available]


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    df = load_dataset(args.dataset)
    hotspots = build_hotspots(df, args.top_safety, args.top_telemetry)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    hotspots.to_csv(output_path, index=False)

    print(f'Hotspots PIA guardados en {_rel(output_path)} (filas={len(hotspots)})')
    write_metadata(
        output_path,
        script=__file__,
        inputs=[args.dataset],
        extra={'rows': int(len(hotspots)), 'columns': int(len(hotspots.columns))},
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
