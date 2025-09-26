#!/usr/bin/env python3
"""Muestra alertas básicas del resumen de planes (synthetic lab)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

DEFAULT_SUMMARY = Path('reports/pia_plan_summary.csv')
DEFAULT_FEATURES = Path('data/hase/pia_outcomes_features.csv')


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Archivo no encontrado: {path}")
    return pd.read_csv(path)


def main(summary: Path = DEFAULT_SUMMARY, features: Path = DEFAULT_FEATURES) -> int:
    summary_df = load_csv(summary)
    features_df = load_csv(features)

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

    print('
=== Alertas ===')
    _report(negatives, 'con protecciones negativas')
    _report(expirados, 'con plan expirado')
    _report(manual, 'marcados para revisión manual')
    return 0


if __name__ == '__main__':
    summary_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SUMMARY
    features_path = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_FEATURES
    raise SystemExit(main(summary_path, features_path))
