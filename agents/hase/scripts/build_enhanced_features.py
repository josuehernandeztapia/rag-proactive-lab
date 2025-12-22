#!/usr/bin/env python3
"""Build enhanced HASE default prediction using original GNV logic + telemetry enrichment.

Usage:
    python agents/hase/scripts/build_enhanced_features.py
    python agents/hase/scripts/build_enhanced_features.py --config config/hase.yml

Genera predicción de default enriquecida:
- CORE (70%): Lógica original basada en consumo GNV y patrones financieros
- ENHANCEMENT (30%): Señales de telemetría como behavioral risk indicators
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd
import yaml
from datetime import datetime, timedelta

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.metadata import write_metadata


def load_config(config_path: Path) -> Dict[str, Any]:
    """Carga configuración YAML."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_telemetry_data(config: Dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carga datos de telemetría enriquecida."""
    paths = config['paths']

    # Cargar telemetría base
    trips_df = pd.read_csv(ROOT / paths['trips_daily'])
    events_df = pd.read_csv(ROOT / paths['events_daily'])

    # Telemetría legacy si existe
    telemetry_path = ROOT / paths.get('telemetry_summary')
    telemetry_df = pd.DataFrame()
    if telemetry_path.exists():
        telemetry_df = pd.read_csv(telemetry_path)

    return trips_df, events_df, telemetry_df


def build_core_gnv_proxy_features(trips_df: pd.DataFrame, events_df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Genera features core que aproximan la lógica original de HASE sin datos GNV.

    ORIGINAL: litros_diarios, recaudo_diario, coverage_ratio_*
    PROXY: engine_hours (consumo), distance_patterns (actividad), usage_consistency (coverage)
    """

    features = []

    # Agregar trips por placa primero
    trips_agg = trips_df.groupby('placa').agg({
        'engine_hours': 'sum',
        'distance_km': 'sum',
        'driving_hours': 'sum',
        'idling_hours': 'sum'
    }).reset_index()

    # Merge con eventos agregados por placa
    merged_df = trips_agg.merge(
        events_df.groupby('placa').agg({
            'idling': 'mean',
            'harsh_brake': 'mean',
            'harsh_maneuver': 'mean',
            'overspeed': 'mean'
        }).reset_index(),
        on='placa', how='left'
    ).fillna(0)

    for _, row in merged_df.iterrows():
        placa = row['placa']

        # CORE FEATURES (Proxy de lógica original HASE)
        engine_hours = max(row.get('engine_hours', 1), 0.1)
        distance = row.get('distance_km', 0)

        # Proxy para litros_diarios (engine_hours como consumo)
        daily_consumption_proxy = engine_hours / max(1, len(merged_df[merged_df['placa'] == placa]))

        # Proxy para coverage_ratio (actividad vs esperado)
        expected_daily_distance = 50  # km baseline
        activity_coverage = min(distance / (expected_daily_distance * 30), 1.0)  # Normalizar a mes

        # Proxy para recaudo_diario (eficiencia como proxy de ingresos)
        operational_efficiency = distance / engine_hours if engine_hours > 0 else 0
        revenue_proxy = operational_efficiency * 100  # Proxy de ingresos basado en eficiencia

        # Default risk score base (lógica original)
        # Alta variabilidad consumo + baja cobertura + baja eficiencia = riesgo default
        consumption_consistency = 1.0 - min(abs(daily_consumption_proxy - 8) / 8, 1.0)  # 8h baseline
        coverage_stability = activity_coverage
        efficiency_stability = min(operational_efficiency / 30, 1.0)  # 30 km/h baseline

        # Core default risk (0-1, higher = more default risk)
        core_default_risk = 1.0 - np.mean([consumption_consistency, coverage_stability, efficiency_stability])

        features.append({
            'placa': placa,
            'daily_consumption_proxy': daily_consumption_proxy,
            'activity_coverage': activity_coverage,
            'revenue_proxy': revenue_proxy,
            'core_default_risk': core_default_risk,
            'consumption_consistency': consumption_consistency,
            'coverage_stability': coverage_stability,
            'efficiency_stability': efficiency_stability,
            'total_distance': distance,
            'total_engine_hours': engine_hours
        })

    return pd.DataFrame(features)


def build_telemetry_behavioral_risk_features(trips_df: pd.DataFrame, events_df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Genera features de riesgo behavioral usando telemetría (ENHANCEMENT 30%).

    Señales que correlacionan con mayor probabilidad de default:
    - after_hours_events → Uso no autorizado → Evasión pagos
    - harsh_driving → Costos operativos altos → Stress financiero
    - device_disconnect → Fraud patterns → Default risk
    - operational_inconsistency → Business instability → Payment issues
    """

    features = []

    # Agregar trips por placa primero
    trips_agg = trips_df.groupby('placa').agg({
        'engine_hours': 'sum',
        'distance_km': 'sum',
        'driving_hours': 'sum',
        'idling_hours': 'sum'
    }).reset_index()

    # Merge con eventos agregados por placa
    merged_df = trips_agg.merge(
        events_df.groupby('placa').agg({
            'idling': 'mean',
            'overspeed': 'mean',
            'harsh_brake': 'mean',
            'harsh_maneuver': 'mean',
            'after_hours': 'mean'
        }).reset_index(),
        on='placa', how='left'
    ).fillna(0)

    for _, row in merged_df.iterrows():
        # BEHAVIORAL RISK SIGNALS (Telemetría → Default Risk)

        # 1. Unauthorized usage risk → Payment avoidance patterns
        after_hours_events = row.get('after_hours', 0)
        unauthorized_usage_risk = min(after_hours_events / 10, 1.0)  # Normalize to max 10 events

        # 2. Operational stress risk → High costs → Financial pressure
        harsh_brake_events = row.get('harsh_brake', 0)
        harsh_maneuver_events = row.get('harsh_maneuver', 0)
        total_harsh_events = harsh_brake_events + harsh_maneuver_events
        operational_stress_risk = min(total_harsh_events / 50, 1.0)  # Normalize to max 50 events

        # 3. Fraud/tampering risk → Payment evasion patterns
        disconnect_events = row.get('device_disconnect', 0)
        fraud_risk = min(disconnect_events / 5, 1.0)  # Normalize to max 5 disconnects

        # 4. Compliance risk → Regulatory costs → Financial burden
        overspeed_events = row.get('overspeed', 0)
        compliance_risk = min(overspeed_events / 20, 1.0)  # Normalize to max 20 violations

        # 5. Operational inconsistency → Business instability
        engine_hours = max(row.get('engine_hours', 1), 0.1)
        distance = max(row.get('distance_km', 1), 1)
        efficiency = distance / engine_hours
        expected_efficiency = 25  # km/h baseline
        inconsistency_risk = min(abs(efficiency - expected_efficiency) / expected_efficiency, 1.0)

        # BEHAVIORAL DEFAULT RISK (composite score)
        behavioral_risk_components = [
            unauthorized_usage_risk * 0.25,
            operational_stress_risk * 0.25,
            fraud_risk * 0.20,
            compliance_risk * 0.15,
            inconsistency_risk * 0.15
        ]

        behavioral_default_risk = sum(behavioral_risk_components)

        features.append({
            'placa': row['placa'],
            'behavioral_default_risk': behavioral_default_risk,
            'unauthorized_usage_risk': unauthorized_usage_risk,
            'operational_stress_risk': operational_stress_risk,
            'fraud_risk': fraud_risk,
            'compliance_risk': compliance_risk,
            'inconsistency_risk': inconsistency_risk,
            'total_harsh_events': total_harsh_events,
            'total_distance': distance,
            'total_engine_hours': engine_hours
        })

    return pd.DataFrame(features)


def build_hase_enhanced_default_prediction(core_df: pd.DataFrame, behavioral_df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Genera predicción de default enriquecida combinando lógica original + telemetría.

    HYBRID SCORING:
    - Core GNV Logic (70%): Consumption patterns + coverage + efficiency
    - Behavioral Enhancement (30%): Telemetry risk signals
    """

    # Merge core features (GNV proxy) + behavioral features (telemetry)
    combined_df = core_df.merge(behavioral_df, on='placa', how='outer').fillna(0)

    default_predictions = []

    for _, row in combined_df.iterrows():
        # HYBRID DEFAULT PREDICTION (según documentación)
        core_risk = row['core_default_risk']  # 0-1 (higher = more risk)
        behavioral_risk = row['behavioral_default_risk']  # 0-1 (higher = more risk)

        # Weighted combination (Core 70% + Behavioral 30%)
        enhanced_default_risk = (core_risk * 0.7) + (behavioral_risk * 0.3)

        # Default probability threshold (configurable)
        default_threshold = config.get('default_prediction', {}).get('default_threshold', 0.6)
        default_flag = 1 if enhanced_default_risk > default_threshold else 0

        # Risk category
        if enhanced_default_risk >= 0.8:
            risk_category = 'high_risk'
            label_reason = 'behavioral_and_consumption_risk'
        elif enhanced_default_risk >= 0.6:
            risk_category = 'medium_risk'
            label_reason = 'consumption_pattern_risk'
        elif enhanced_default_risk >= 0.4:
            risk_category = 'moderate_risk'
            label_reason = 'behavioral_signals'
        else:
            risk_category = 'low_risk'
            label_reason = 'stable'

        # Risk factor attribution
        risk_factors = []
        if core_risk > 0.5:
            risk_factors.append('consumption_patterns')
        if row['unauthorized_usage_risk'] > 0.3:
            risk_factors.append('unauthorized_usage')
        if row['fraud_risk'] > 0.3:
            risk_factors.append('device_tampering')
        if row['operational_stress_risk'] > 0.3:
            risk_factors.append('harsh_driving')

        default_predictions.append({
            'placa': row['placa'],
            'default_flag': default_flag,
            'enhanced_default_risk': enhanced_default_risk,
            'core_default_risk': core_risk,
            'behavioral_default_risk': behavioral_risk,
            'risk_category': risk_category,
            'label_reason': label_reason,
            'risk_factors': ';'.join(risk_factors) if risk_factors else 'stable_profile',
            'confidence': abs(enhanced_default_risk - 0.5) * 2,  # Distance from uncertainty

            # Core components breakdown
            'consumption_consistency': row['consumption_consistency'],
            'coverage_stability': row['coverage_stability'],
            'activity_coverage': row['activity_coverage'],

            # Behavioral components breakdown
            'unauthorized_usage_risk': row['unauthorized_usage_risk'],
            'fraud_risk': row['fraud_risk'],
            'operational_stress_risk': row['operational_stress_risk']
        })

    # Convert to DataFrame
    hase_df = pd.DataFrame(default_predictions)

    # Calculate risk rankings
    hase_df['risk_rank'] = hase_df['enhanced_default_risk'].rank(method='dense', ascending=False)

    return hase_df


def generate_hase_default_alerts(hase_df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Genera alertas y recomendaciones basadas en predicción de default."""

    alerts = []
    alert_config = config.get('alerts', {})

    for _, row in hase_df.iterrows():
        placa = row['placa']
        default_risk = row['enhanced_default_risk']
        risk_category = row['risk_category']

        # Generate alerts based on default risk
        if default_risk >= 0.8:
            alert_type = 'default_imminent'
            priority = 'critical'
            recommendation = 'Acción inmediata: Revisión de crédito y reestructuración'
        elif default_risk >= 0.6:
            alert_type = 'default_risk_high'
            priority = 'high'
            recommendation = 'Monitoreo estrecho y ajuste de límites de crédito'
        elif default_risk >= 0.4:
            alert_type = 'default_risk_moderate'
            priority = 'medium'
            recommendation = 'Seguimiento de patrones y alertas tempranas'
        else:
            alert_type = 'default_risk_low'
            priority = 'low'
            recommendation = 'Mantener monitoreo rutinario'

        # Risk factor analysis
        risk_factors_list = row['risk_factors'].split(';') if row['risk_factors'] != 'stable_profile' else []

        # Specific recommendations based on risk factors
        action_items = []
        if 'unauthorized_usage' in risk_factors_list:
            action_items.append('Implementar controles de uso autorizado')
        if 'device_tampering' in risk_factors_list:
            action_items.append('Investigar manipulación de dispositivos')
        if 'consumption_patterns' in risk_factors_list:
            action_items.append('Revisar patrones de consumo y eficiencia')
        if 'harsh_driving' in risk_factors_list:
            action_items.append('Programa de capacitación en conducción')

        alerts.append({
            'placa': placa,
            'default_flag': row['default_flag'],
            'enhanced_default_risk': default_risk,
            'risk_category': risk_category,
            'alert_type': alert_type,
            'priority': priority,
            'recommendation': recommendation,
            'action_items': '; '.join(action_items) if action_items else 'Mantener monitoreo',
            'risk_factors': row['risk_factors'],
            'confidence': row['confidence'],
            'generated_at': datetime.now().strftime('%Y-%m-%d'),
            'core_risk_component': row['core_default_risk'],
            'behavioral_risk_component': row['behavioral_default_risk']
        })

    return pd.DataFrame(alerts)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build enhanced HASE features from telemetry")
    parser.add_argument('--config', default='config/hase.yml', help='Path to HASE configuration file')

    args = parser.parse_args(list(argv) if argv is not None else None)

    config_path = ROOT / args.config
    if not config_path.exists():
        print(f"❌ Configuration file not found: {config_path}")
        return 1

    config = load_config(config_path)

    print("🎯 Construyendo HASE Enhanced Default Prediction...")
    print("Core (70%): GNV consumption proxy + coverage patterns")
    print("Enhancement (30%): Behavioral risk signals from telemetry")

    # Cargar datos
    trips_df, events_df, telemetry_df = load_telemetry_data(config)
    print(f"📊 Datos cargados: {len(trips_df)} trips, {len(events_df)} event days")

    # Generar features core (proxy de lógica original)
    print("🔋 Generando core features (proxy lógica GNV original)...")
    core_features = build_core_gnv_proxy_features(trips_df, events_df, config)

    print("💡 Generando behavioral risk features (telemetría enhancement)...")
    behavioral_features = build_telemetry_behavioral_risk_features(trips_df, events_df, config)

    print("⚖️ Combinando en predicción híbrida (70% core + 30% behavioral)...")
    default_predictions = build_hase_enhanced_default_prediction(core_features, behavioral_features, config)

    print("🚨 Generando alertas de riesgo de default...")
    default_alerts = generate_hase_default_alerts(default_predictions, config)

    # Guardar salidas
    paths = config['paths']

    # Core features (proxy GNV logic)
    core_output = ROOT / "data/processed/hase/core_gnv_proxy_features.csv"
    core_output.parent.mkdir(parents=True, exist_ok=True)
    core_features.to_csv(core_output, index=False)

    # Behavioral risk features (telemetry enhancement)
    behavioral_output = ROOT / "data/processed/hase/behavioral_risk_features.csv"
    behavioral_output.parent.mkdir(parents=True, exist_ok=True)
    behavioral_features.to_csv(behavioral_output, index=False)

    # HASE default alerts (main output)
    alerts_output = ROOT / paths['output']
    alerts_output.parent.mkdir(parents=True, exist_ok=True)
    default_alerts.to_csv(alerts_output, index=False)

    # Enhanced default predictions (complete dataset)
    predictions_output = ROOT / "data/processed/hase/enhanced_default_predictions.csv"
    default_predictions.to_csv(predictions_output, index=False)

    # Resumen
    print(f"\n✅ HASE Enhanced Default Prediction generado:")
    print(f"🔋 Core features: {len(core_features)} placas")
    print(f"💡 Behavioral features: {len(behavioral_features)} placas")
    print(f"⚖️ Default predictions: {len(default_predictions)} placas")
    print(f"🚨 Alertas generadas: {len(default_alerts)} alertas")

    # Distribución de riesgo default
    if len(default_predictions) > 0:
        print("\n🎯 Distribución de riesgo de default:")
        risk_dist = default_predictions['risk_category'].value_counts()
        for category, count in risk_dist.items():
            print(f"  - {category}: {count}")

        default_flags = default_predictions['default_flag'].value_counts()
        print(f"\n🚨 Predicciones de default:")
        print(f"  - Default predicted: {default_flags.get(1, 0)}")
        print(f"  - Stable predicted: {default_flags.get(0, 0)}")

        avg_risk = default_predictions['enhanced_default_risk'].mean()
        avg_confidence = default_predictions['confidence'].mean()
        print(f"\n📊 Riesgo promedio: {avg_risk:.3f} (confidence: {avg_confidence:.3f})")

    # Metadata
    write_metadata(
        alerts_output,
        script=__file__,
        inputs=[config_path],
        extra={
            'plates_analyzed': int(len(default_predictions)),
            'default_alerts_generated': int(len(default_alerts)),
            'default_predictions': int(default_predictions['default_flag'].sum()) if len(default_predictions) > 0 else 0,
            'average_default_risk': float(default_predictions['enhanced_default_risk'].mean()) if len(default_predictions) > 0 else 0,
            'risk_distribution': default_predictions['risk_category'].value_counts().to_dict() if len(default_predictions) > 0 else {},
            'core_weight': 0.7,
            'behavioral_weight': 0.3
        }
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())