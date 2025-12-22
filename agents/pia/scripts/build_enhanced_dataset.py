#!/usr/bin/env python3
"""Build enhanced PIA portfolio risk + protection recommendations using original logic + telemetry.

Usage:
    python agents/pia/scripts/build_enhanced_dataset.py
    python agents/pia/scripts/build_enhanced_dataset.py --config config/pia.yml

Genera scoring de riesgo de cartera enriquecido:
- CORE (80%): Lógica original (financial profile + GNV credit + arrears)
- ENHANCEMENT (20%): Señales de telemetría (behavioral + operational risk)
Pesos matemáticamente optimizados basados en validación cuantitativa.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd
import yaml
from datetime import datetime

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.metadata import write_metadata


def load_config(config_path: Path) -> Dict[str, Any]:
    """Carga configuración YAML."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_telemetry_data(config: Dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carga datos de telemetría + original PIA features para enriquecimiento."""
    paths = config['paths']

    # Cargar telemetría
    trips_df = pd.read_csv(ROOT / paths['trips_daily'])
    events_df = pd.read_csv(ROOT / paths['events_daily'])

    # Cargar features originales PIA (financial profile)
    original_pia_features = pd.DataFrame()
    pia_original_path = ROOT / "data/processed/pia/pia_features.csv"
    if pia_original_path.exists():
        original_pia_features = pd.read_csv(pia_original_path)

    # Cargar HASE enhanced features para integración
    hase_features = pd.DataFrame()
    hase_path = ROOT / "data/processed/hase/enhanced_default_predictions.csv"
    if hase_path.exists():
        hase_features = pd.read_csv(hase_path)

    return trips_df, events_df, original_pia_features, hase_features


def build_core_pia_financial_features(original_pia_df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Genera features core PIA basados en lógica original (financial + GNV).

    ORIGINAL PIA: coverage_ratio_*, gnv_credit_30d, arrears_amount, expected_payment
    Mantiene la lógica de negocio de riesgo financiero + protección de cartera.
    """

    # Si no hay features originales, crear proxy basado en baseline
    if original_pia_df.empty:
        print("⚠️  No hay features PIA originales, creando proxy financiero...")
        return pd.DataFrame({
            'placa': ['A-05501-A', 'A-05355-A', 'A-05502-A', 'A-05503-A', 'A-05504-A', 'A-05507-A', 'A-05508-A'],
            'core_financial_risk': [0.5] * 7,  # Placeholder
            'expected_payment': [18000.0] * 7,
            'coverage_ratio_30d': [0.7] * 7,
            'arrears_amount': [5000.0] * 7,
            'gnv_credit_30d': [12000.0] * 7
        })

    # Usar features originales existentes
    core_features = []
    for _, row in original_pia_df.iterrows():
        placa = row['placa']

        # CORE FINANCIAL RISK (lógica original PIA)
        coverage_30d = row.get('coverage_ratio_30d', 0.5)
        gnv_credit = row.get('gnv_credit_30d', 0)
        arrears = row.get('arrears_amount', 0)
        expected_payment = row.get('expected_payment', 18000)

        # Financial stress indicators (original logic)
        coverage_risk = 1.0 - coverage_30d  # Lower coverage = higher risk
        credit_utilization = min(gnv_credit / (expected_payment * 0.8), 1.0)  # Credit vs expected
        arrears_risk = min(arrears / expected_payment, 1.0)  # Arrears vs payment capacity

        # Core financial risk score (original PIA logic)
        core_financial_risk = np.mean([
            coverage_risk * 0.4,      # Coverage es key indicator
            credit_utilization * 0.3,  # Credit utilization
            arrears_risk * 0.3        # Outstanding arrears
        ])

        core_features.append({
            'placa': placa,
            'core_financial_risk': core_financial_risk,
            'coverage_ratio_30d': coverage_30d,
            'gnv_credit_30d': gnv_credit,
            'arrears_amount': arrears,
            'expected_payment': expected_payment,
            'coverage_risk': coverage_risk,
            'credit_utilization': credit_utilization,
            'arrears_risk': arrears_risk
        })

    return pd.DataFrame(core_features)


def build_safety_risk_features(events_df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Genera features de riesgo de seguridad para cartera PIA.

    Riesgo de siniestralidad → Claims → Portfolio loss
    """
    safety_events_config = config.get('enhanced_features', {}).get('safety_events', [])

    safety_features = []

    # Agregar por placa los eventos de seguridad
    safety_agg = events_df.groupby('placa').agg({
        'harsh_brake': 'sum',
        'harsh_maneuver': 'sum',
        'overspeed': 'sum',
        'seatbelt_off': 'sum'
    }).reset_index().fillna(0)

    for _, row in safety_agg.iterrows():
        placa = row['placa']

        # Eventos de seguridad
        harsh_brake_events = row.get('harsh_brake', 0)
        harsh_maneuver_events = row.get('harsh_maneuver', 0)
        overspeed_events = row.get('overspeed', 0)
        seatbelt_events = row.get('seatbelt_off', 0)

        total_safety_events = harsh_brake_events + harsh_maneuver_events + overspeed_events + seatbelt_events

        # Score de riesgo de accidentes (normalizado)
        accident_risk = min((
            harsh_brake_events / 30 * 0.3 +  # 30 eventos harsh_brake baseline
            harsh_maneuver_events / 25 * 0.3 +  # 25 eventos harsh_maneuver baseline
            overspeed_events / 50 * 0.25 +  # 50 eventos overspeed baseline
            seatbelt_events / 20 * 0.15  # 20 eventos seatbelt baseline
        ), 1.0)

        # Frecuencias normalizadas
        harsh_brake_frequency = min(harsh_brake_events / 30, 1.0)
        harsh_maneuver_frequency = min(harsh_maneuver_events / 25, 1.0)
        overspeed_frequency = min(overspeed_events / 50, 1.0)
        seatbelt_frequency = min(seatbelt_events / 20, 1.0)

        # Riesgo final de seguridad
        final_safety_risk = accident_risk

        safety_features.append({
            'placa': placa,
            'harsh_brake_events': harsh_brake_events,
            'harsh_maneuver_events': harsh_maneuver_events,
            'overspeed_events': overspeed_events,
            'seatbelt_events': seatbelt_events,
            'total_safety_events': total_safety_events,
            'harsh_brake_frequency': harsh_brake_frequency,
            'harsh_maneuver_frequency': harsh_maneuver_frequency,
            'overspeed_frequency': overspeed_frequency,
            'seatbelt_frequency': seatbelt_frequency,
            'accident_risk_score': accident_risk,
            'final_safety_risk': final_safety_risk
        })

    return pd.DataFrame(safety_features)


def build_operational_risk_features(trips_df: pd.DataFrame, events_df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Genera features de riesgo operacional para cartera PIA.

    Señales operacionales que afectan riesgo de cartera:
    - fraud_risk → Asset manipulation → Portfolio exposure
    - compliance_risk → Regulatory costs → Portfolio liability
    - operational_inconsistency → Business instability → Portfolio risk
    """

    operational_features = []

    # Merge trips con eventos operacionales
    operational_agg = events_df.groupby('placa').agg({
        'idling': 'sum',
        'pto': 'sum',
        'device_disconnect': 'sum',
        'after_hours': 'sum'
    }).reset_index().fillna(0)

    trips_agg = trips_df.groupby('placa').agg({
        'distance_km': 'sum',
        'engine_hours': 'sum'
    }).reset_index().fillna(0)

    merged_df = operational_agg.merge(trips_agg, on='placa', how='outer').fillna(0)

    for _, row in merged_df.iterrows():
        placa = row['placa']

        # Eficiencia de combustible proxy (distance/engine_hours)
        distance = row.get('distance_km', 0)
        engine_hours = max(row.get('engine_hours', 1), 0.1)
        fuel_efficiency_proxy = distance / engine_hours

        # Riesgo por ralentí excesivo
        idling_events = row.get('idling', 0)
        idle_risk = min(idling_events / 50, 1.0)  # Normalizar a 50 eventos

        # Riesgo por uso no autorizado
        after_hours_events = row.get('after_hours', 0)
        unauthorized_use_risk = min(after_hours_events / 10, 1.0)

        # Riesgo por disconnections (fraude/manipulación)
        disconnect_events = row.get('device_disconnect', 0)
        tampering_risk = min(disconnect_events / 5, 1.0)

        # PTO risk (uso no autorizado de equipos)
        pto_events = row.get('pto', 0)
        pto_risk = min(pto_events / 3, 1.0)

        # Score compuesto de riesgo operacional
        operational_risk_score = np.mean([
            idle_risk,
            unauthorized_use_risk,
            tampering_risk,
            pto_risk
        ])

        # Factor de eficiencia (menor eficiencia = mayor riesgo)
        efficiency_factor = max(0, 1.0 - (fuel_efficiency_proxy / 50))  # 50 km/h como baseline

        # Score final
        final_operational_risk = np.mean([operational_risk_score, efficiency_factor])

        operational_features.append({
            'placa': placa,
            'fuel_efficiency_proxy': fuel_efficiency_proxy,
            'idle_risk': idle_risk,
            'unauthorized_use_risk': unauthorized_use_risk,
            'tampering_risk': tampering_risk,
            'pto_risk': pto_risk,
            'efficiency_factor': efficiency_factor,
            'operational_risk_score': operational_risk_score,
            'final_operational_risk': final_operational_risk,
            'total_distance': distance,
            'total_engine_hours': engine_hours,
            'idling_events': idling_events,
            'after_hours_events': after_hours_events,
            'disconnect_events': disconnect_events,
            'pto_events': pto_events
        })

    return pd.DataFrame(operational_features)


def build_combined_risk_model(financial_df: pd.DataFrame, safety_df: pd.DataFrame, operational_df: pd.DataFrame, hase_df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Combina features financieros core PIA + telemetría en modelo híbrido.

    LÓGICA PIA ENRIQUECIDA (MATEMÁTICAMENTE OPTIMIZADA):
    - CORE (80%): Lógica original PIA (financial profile + GNV credit + arrears)
    - ENHANCEMENT (20%): Señales telemetría (safety + operational risk)

    Pesos ajustados basado en validación matemática que demostró:
    - Independencia excelente entre componentes (r=-0.071)
    - Core financiero dominante en contribución (r=0.877 vs r=0.088)
    - Optimización aumenta discriminación en +18%
    """

    # Merge core financial features con telemetría
    combined_df = financial_df.merge(safety_df, on='placa', how='outer')
    combined_df = combined_df.merge(operational_df, on='placa', how='outer')

    if not hase_df.empty:
        # Incluir features de HASE si están disponibles (enhanced default predictions)
        hase_risk_cols = ['placa', 'enhanced_default_risk', 'risk_category']
        available_cols = [col for col in hase_risk_cols if col in hase_df.columns]
        if len(available_cols) >= 2:  # Al menos placa + 1 risk feature
            combined_df = combined_df.merge(
                hase_df[available_cols],
                on='placa', how='left'
            )
        else:
            combined_df['enhanced_default_risk'] = 0.5
            combined_df['risk_category'] = 'moderate_risk'
    else:
        combined_df['enhanced_default_risk'] = 0.5
        combined_df['risk_category'] = 'moderate_risk'

    combined_df = combined_df.fillna(0)

    # Weights from config
    weights = config.get('risk_scoring', {}).get('weights', {})

    risk_features = []

    for _, row in combined_df.iterrows():
        # LÓGICA HÍBRIDA PIA ENRIQUECIDA

        # CORE FINANCIERO (60% - Lógica original PIA)
        core_financial_risk = row.get('core_financial_risk', 0.5)

        # ENHANCEMENT TELEMETRÍA (40% - Safety + Operational)
        safety_risk_component = row.get('final_safety_risk', 0)
        operational_risk_component = row.get('final_operational_risk', 0)

        # Combinar components telemetría
        telemetry_enhancement_score = np.mean([
            safety_risk_component,  # 50% safety
            operational_risk_component  # 50% operational
        ])

        # SCORING HÍBRIDO FINAL (MATEMÁTICAMENTE OPTIMIZADO)
        # Validación matemática recomienda 80/20 para mejor discriminación
        # Ver: docs/VALIDACION_MATEMATICA_PESOS_HIBRIDOS.md
        overall_portfolio_risk = (
            core_financial_risk * 0.8 +  # Core PIA logic (optimizado)
            telemetry_enhancement_score * 0.2  # Telemetry signals (complementario)
        )

        # Para compatibilidad con configuración existente, mantener overall_risk_score
        overall_risk_score = overall_portfolio_risk

        # Risk category
        thresholds = config.get('risk_scoring', {}).get('thresholds', {})
        if overall_risk_score >= thresholds.get('high_risk', 0.75):
            risk_category = 'high'
        elif overall_risk_score >= thresholds.get('medium_risk', 0.50):
            risk_category = 'medium'
        else:
            risk_category = 'low'

        # Projected insurance cost (synthetic target)
        base_cost = config.get('synthetic_data', {}).get('target_payment', 18000)

        # Risk multiplier
        if risk_category == 'high':
            cost_multiplier = 1.5 + (overall_risk_score - 0.75) * 2.0
        elif risk_category == 'medium':
            cost_multiplier = 1.0 + (overall_risk_score - 0.25) * 1.0
        else:
            cost_multiplier = 0.7 + overall_risk_score * 0.6

        projected_cost = base_cost * cost_multiplier

        risk_features.append({
            'placa': row['placa'],
            # SCORING HÍBRIDO PIA ENRIQUECIDO
            'overall_portfolio_risk': overall_portfolio_risk,
            'core_financial_risk': core_financial_risk,
            'telemetry_enhancement_score': telemetry_enhancement_score,
            'overall_risk_score': overall_risk_score,  # Para compatibilidad
            'risk_category': risk_category,
            'projected_insurance_cost': projected_cost,
            # COMPONENTES CORE FINANCIERO
            'coverage_ratio_30d': row.get('coverage_ratio_30d', 0.5),
            'gnv_credit_30d': row.get('gnv_credit_30d', 0),
            'arrears_amount': row.get('arrears_amount', 0),
            'expected_payment': row.get('expected_payment', 18000),
            'coverage_risk': row.get('coverage_risk', 0.5),
            'credit_utilization': row.get('credit_utilization', 0),
            'arrears_risk': row.get('arrears_risk', 0),
            # COMPONENTES TELEMETRÍA
            'safety_risk_component': safety_risk_component,
            'operational_risk_component': operational_risk_component,
            'accident_risk_score': row.get('accident_risk_score', 0),
            'fuel_efficiency_proxy': row.get('fuel_efficiency_proxy', 0),
            'unauthorized_use_risk': row.get('unauthorized_use_risk', 0),
            'tampering_risk': row.get('tampering_risk', 0),
            # EVENTOS AGREGADOS
            'total_safety_events': row.get('total_safety_events', 0),
            'harsh_brake_events': row.get('harsh_brake_events', 0),
            'overspeed_events': row.get('overspeed_events', 0),
            'idling_events': row.get('idling_events', 0),
            'after_hours_events': row.get('after_hours_events', 0),
            'disconnect_events': row.get('disconnect_events', 0),
            # INTEGRACIÓN HASE
            'hase_default_risk': row.get('enhanced_default_risk', 0.5)
        })

    return pd.DataFrame(risk_features)


def generate_synthetic_scenarios(risk_df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Genera escenarios sintéticos para entrenamiento."""

    synthetic_config = config.get('synthetic_data', {})
    scenarios = synthetic_config.get('scenarios', {})

    synthetic_data = []
    base_df = risk_df.copy()

    # Escenario conservador (30%)
    conservative_count = int(len(base_df) * scenarios.get('conservative_driver', 0.3))
    conservative_drivers = base_df.sample(n=conservative_count, random_state=42).copy()

    for _, driver in conservative_drivers.iterrows():
        # Reducir riesgo para conductores conservadores
        original_risk = driver.get('overall_portfolio_risk', driver.get('overall_risk_score', 0.5))
        synthetic_data.append({
            **driver.to_dict(),
            'overall_portfolio_risk': original_risk * 0.6,
            'overall_risk_score': original_risk * 0.6,
            'risk_category': 'low',
            'projected_insurance_cost': driver['projected_insurance_cost'] * 0.8,
            'scenario_type': 'conservative'
        })

    # Escenario agresivo (20%)
    aggressive_count = int(len(base_df) * scenarios.get('aggressive_driver', 0.2))
    aggressive_drivers = base_df.sample(n=aggressive_count, random_state=43).copy()

    for _, driver in aggressive_drivers.iterrows():
        # Aumentar riesgo para conductores agresivos
        original_risk = driver.get('overall_portfolio_risk', driver.get('overall_risk_score', 0.5))
        enhanced_risk = min(original_risk * 1.8, 1.0)
        synthetic_data.append({
            **driver.to_dict(),
            'overall_portfolio_risk': enhanced_risk,
            'overall_risk_score': enhanced_risk,
            'risk_category': 'high',
            'projected_insurance_cost': driver['projected_insurance_cost'] * 1.6,
            'scenario_type': 'aggressive'
        })

    # Comportamiento mixto (50% restante usa datos reales)
    return pd.concat([risk_df, pd.DataFrame(synthetic_data)], ignore_index=True)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build enhanced PIA risk dataset")
    parser.add_argument('--config', default='config/pia.yml', help='Path to PIA configuration file')

    args = parser.parse_args(list(argv) if argv is not None else None)

    config_path = ROOT / args.config
    if not config_path.exists():
        print(f"❌ Configuration file not found: {config_path}")
        return 1

    config = load_config(config_path)

    print("🎯 Construyendo dataset PIA enriquecido...")

    # Cargar datos
    trips_df, events_df, original_pia_features, hase_df = load_telemetry_data(config)
    print(f"📊 Datos cargados: {len(trips_df)} trips, {len(events_df)} event days, {len(original_pia_features)} PIA original features, {len(hase_df)} HASE features")

    # Generar features core PIA (lógica original)
    print("💰 Generando features financieros core PIA...")
    financial_features = build_core_pia_financial_features(original_pia_features, config)

    # Generar features enriquecidos con telemetría
    print("🚨 Generando features de riesgo de seguridad...")
    safety_features = build_safety_risk_features(events_df, config)

    print("🔧 Generando features de riesgo operacional...")
    operational_features = build_operational_risk_features(trips_df, events_df, config)

    print("🎯 Combinando en modelo de riesgo de cartera híbrido...")
    risk_model = build_combined_risk_model(financial_features, safety_features, operational_features, hase_df, config)

    print("🔄 Generando escenarios sintéticos...")
    enhanced_dataset = generate_synthetic_scenarios(risk_model, config)

    # Guardar salidas
    paths = config['paths']

    # Dataset enriquecido
    output_path = ROOT / paths['output']
    output_path.parent.mkdir(parents=True, exist_ok=True)
    enhanced_dataset.to_csv(output_path, index=False)

    # Features individuales
    safety_path = ROOT / "data/processed/pia/safety_features_enhanced.csv"
    safety_features.to_csv(safety_path, index=False)

    operational_path = ROOT / "data/processed/pia/operational_features_enhanced.csv"
    operational_features.to_csv(operational_path, index=False)

    # Resumen
    print(f"\n✅ PIA dataset enriquecido generado:")
    print(f"🎯 Registros totales: {len(enhanced_dataset)}")
    print(f"📊 Features de seguridad: {len(safety_features)} placas")
    print(f"🔧 Features operacionales: {len(operational_features)} placas")

    # Distribución de riesgo
    risk_distribution = enhanced_dataset['risk_category'].value_counts()
    print(f"\n📊 Distribución de riesgo:")
    for category, count in risk_distribution.items():
        percentage = (count / len(enhanced_dataset)) * 100
        print(f"  - {category}: {count} ({percentage:.1f}%)")

    # Estadísticas de costo proyectado
    avg_cost = enhanced_dataset['projected_insurance_cost'].mean()
    print(f"\n💰 Costo promedio proyectado: ${avg_cost:,.0f}")

    # Metadata
    write_metadata(
        output_path,
        script=__file__,
        inputs=[config_path],
        extra={
            'placas_processed': int(len(risk_model)),
            'synthetic_scenarios': int(len(enhanced_dataset) - len(risk_model)),
            'risk_distribution': risk_distribution.to_dict(),
            'average_projected_cost': float(avg_cost)
        }
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())