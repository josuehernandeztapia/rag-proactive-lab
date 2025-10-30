#!/usr/bin/env python3
"""Muestra alertas básicas del resumen de planes (synthetic lab)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.loader import get_path

DEFAULT_SUMMARY = get_path('reports', 'pia_plan_summary')
DEFAULT_FEATURES = get_path('data', 'processed', 'hase', 'outcomes_features')
DEFAULT_PIA_DATASET = get_path('data', 'processed', 'pia', 'features_augmented')


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Archivo no encontrado: {_rel(path)}")
    return pd.read_csv(path)


def main(summary: Path = DEFAULT_SUMMARY, features: Path = DEFAULT_FEATURES, pia_features: Path = DEFAULT_PIA_DATASET) -> int:
    summary_df = load_csv(summary)
    features_df = load_csv(features)
    try:
        pia_df = load_csv(pia_features)
    except SystemExit:
        pia_df = pd.DataFrame()

    print('=== Resumen por plan ===')
    print(summary_df.to_string(index=False))

    negatives = features_df[features_df.get('protections_flag_negative', False)]
    expirados = features_df[features_df.get('protections_flag_expired', False)]
    manual = features_df[features_df.get('protections_flag_manual', False)]

    def _report(df: pd.DataFrame, label: str) -> None:
        if df.empty:
            print(f'No hay contratos {label}.')
        else:
            print(f'Contratos {label}: {len(df)}')
            print(df[['placa', 'last_plan_type', 'last_plan_status', 'protections_remaining']].to_string(index=False))

    print('\n=== Alertas ===')
    _report(negatives, 'con protecciones negativas')
    _report(expirados, 'con plan expirado')
    _report(manual, 'marcados para revisión manual')

    if not pia_df.empty:
        desired = ['placa', 'plaza_limpia', 'safety_alert', 'telemetry_alert', 'seatbelt_off_rate_30d', 'high_speed_ratio_30d', 'telemetry_health_score', 'idle_hours_ratio_30d']
        available = [col for col in desired if col in pia_df.columns]
        merged = pia_df[available].copy()
        if 'placa' not in merged.columns and 'placa' in pia_df.columns:
            merged['placa'] = pia_df['placa']
        if 'plaza_limpia' not in merged.columns and 'plaza' in pia_df.columns:
            merged['plaza_limpia'] = pia_df['plaza']
        merged['safety_alert'] = merged.get('safety_alert', 0).fillna(0)
        merged['telemetry_alert'] = merged.get('telemetry_alert', 0).fillna(0)

        safety = merged[merged['safety_alert'] == 1]
        telemetry = merged[merged['telemetry_alert'] == 1]

        print('\n=== Seguridad / Telemetría ===')
        print(f"Placas con safety_alert=1: {len(safety)}")
        if not safety.empty:
            print(safety.sort_values('seatbelt_off_rate_30d', ascending=False).head(10)[
                ['placa', 'plaza_limpia', 'seatbelt_off_rate_30d', 'high_speed_ratio_30d']
            ].to_string(index=False))
        print(f"\nPlacas con telemetry_alert=1: {len(telemetry)}")
        if not telemetry.empty:
            print(telemetry.sort_values('telemetry_health_score').head(10)[
                ['placa', 'plaza_limpia', 'telemetry_health_score', 'idle_hours_ratio_30d']
            ].to_string(index=False))
    else:
        print('\n(No se encontró el dataset PIA, omitiendo sección de seguridad/telemetría)')
    return 0


if __name__ == '__main__':
    summary_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SUMMARY
    features_path = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_FEATURES
    pia_path = Path(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_PIA_DATASET
    raise SystemExit(main(summary_path, features_path, pia_path))
