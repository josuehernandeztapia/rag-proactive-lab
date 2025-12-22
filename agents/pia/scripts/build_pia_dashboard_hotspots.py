#!/usr/bin/env python3
"""Genera un feed curado con hotspots de riesgo usando features enriquecidos PIA.

USA SCORING HÍBRIDO OPTIMIZADO:
- overall_portfolio_risk (80% financial + 20% telemetría)
- core_financial_risk (component financiero puro)
- telemetry_enhancement_score (component telemetría puro)
- safety_risk_component (granular safety signals)
- operational_risk_component (granular operational signals)
"""
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

DEFAULT_DATASET = PROJECT_ROOT / 'data' / 'processed' / 'pia' / 'pia_features_enhanced.csv'
DEFAULT_OUTPUT = get_path('data', 'processed', 'pia', 'hotspots')


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Construye el feed curado de hotspots PIA')
    parser.add_argument('--dataset', type=Path, default=DEFAULT_DATASET, help='Dataset PIA (pia_features_enhanced)')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT, help='Archivo CSV de salida')
    parser.add_argument('--top-portfolio-risk', type=int, default=50, help='Placas con mayor riesgo de cartera')
    parser.add_argument('--top-financial-risk', type=int, default=30, help='Placas con mayor riesgo financiero')
    parser.add_argument('--top-telemetry-risk', type=int, default=30, help='Placas con mayor riesgo telemetría')
    return parser.parse_args(argv)


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f'Dataset no encontrado: {_rel(path)}')
    df = pd.read_csv(path)
    # Verificar features enriquecidos
    required_enhanced = {'placa', 'overall_portfolio_risk', 'core_financial_risk', 'telemetry_enhancement_score'}
    missing = required_enhanced - set(df.columns)
    if missing:
        raise SystemExit(f'Dataset PIA enriquecido carece de columnas requeridas: {sorted(missing)}')
    return df


def build_hotspots_enhanced(df: pd.DataFrame, top_portfolio: int, top_financial: int, top_telemetry: int) -> pd.DataFrame:
    """Construye hotspots usando features enriquecidos PIA."""
    dfs: list[pd.DataFrame] = []

    # Top Portfolio Risk (scoring híbrido optimizado)
    portfolio_df = df.nlargest(top_portfolio, 'overall_portfolio_risk').copy()
    portfolio_df['hotspot_type'] = 'portfolio_risk'
    portfolio_df['priority'] = 'HIGH'
    dfs.append(portfolio_df)

    # Top Financial Risk (component core)
    financial_df = df.nlargest(top_financial, 'core_financial_risk').copy()
    financial_df['hotspot_type'] = 'financial_risk'
    financial_df['priority'] = 'MEDIUM'
    dfs.append(financial_df)

    # Top Telemetry Risk (component enhancement)
    if 'telemetry_enhancement_score' in df.columns:
        telemetry_df = df[df['telemetry_enhancement_score'] > 0].nlargest(top_telemetry, 'telemetry_enhancement_score').copy()
        if not telemetry_df.empty:
            telemetry_df['hotspot_type'] = 'telemetry_risk'
            telemetry_df['priority'] = 'MEDIUM'
            dfs.append(telemetry_df)

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    # Usar features enriquecidos
    columns = [
        'hotspot_type',
        'priority',
        'placa',
        'overall_portfolio_risk',        # Scoring híbrido optimizado
        'core_financial_risk',           # Component financiero
        'telemetry_enhancement_score',   # Component telemetría
        'safety_risk_component',         # Granular safety
        'operational_risk_component',    # Granular operational
        'risk_category',
        'projected_insurance_cost',
        'coverage_ratio_30d',           # Core financial features
        'arrears_amount',
        'harsh_brake_events',           # Telemetry details
        'overspeed_events',
        'idling_events',
        'after_hours_events'
    ]
    available = [col for col in columns if col in combined.columns]
    return combined[available]


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    df = load_dataset(args.dataset)
    hotspots = build_hotspots_enhanced(df, args.top_portfolio_risk, args.top_financial_risk, args.top_telemetry_risk)

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
