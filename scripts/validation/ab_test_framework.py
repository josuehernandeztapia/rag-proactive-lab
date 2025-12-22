#!/usr/bin/env python3
"""Framework de A/B Testing para validar pesos optimizados en producción.

Usage:
    python scripts/validation/ab_test_framework.py --setup
    python scripts/validation/ab_test_framework.py --assign-traffic
    python scripts/validation/ab_test_framework.py --analyze-results

Implementa split traffic para comparar pesos 60/40 vs 80/20 en environment real.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

AB_TEST_CONFIG_PATH = ROOT / "config/ab_test_config.json"
AB_TEST_RESULTS_PATH = ROOT / "data/validation/ab_test_results"


def setup_ab_test(test_name: str = "pia_weights_optimization", duration_days: int = 14) -> Dict[str, Any]:
    """Configura A/B test para validar pesos optimizados."""

    print(f"🧪 CONFIGURANDO A/B TEST: {test_name}")

    # Crear directorio si no existe
    AB_TEST_RESULTS_PATH.mkdir(parents=True, exist_ok=True)

    ab_config = {
        'test_name': test_name,
        'start_date': datetime.now().isoformat(),
        'end_date': (datetime.now() + timedelta(days=duration_days)).isoformat(),
        'status': 'ACTIVE',
        'variants': {
            'control': {
                'name': 'Pesos Actuales 60/40',
                'description': 'Implementación actual PIA',
                'config': {
                    'core_weight': 0.6,
                    'telemetry_weight': 0.4
                },
                'traffic_split': 50,  # 50% del tráfico
                'expected_performance': 'baseline'
            },
            'treatment': {
                'name': 'Pesos Optimizados 80/20',
                'description': 'Pesos matemáticamente optimizados',
                'config': {
                    'core_weight': 0.8,
                    'telemetry_weight': 0.2
                },
                'traffic_split': 50,  # 50% del tráfico
                'expected_performance': '+18% discriminación'
            }
        },
        'metrics': {
            'primary': [
                'risk_score_accuracy',
                'risk_category_precision',
                'portfolio_prediction_accuracy'
            ],
            'secondary': [
                'processing_time',
                'user_satisfaction',
                'false_positive_rate',
                'false_negative_rate'
            ]
        },
        'success_criteria': {
            'minimum_sample_size': 1000,
            'minimum_improvement': 0.05,  # 5% mejora mínima
            'statistical_significance': 0.95,
            'power': 0.80
        },
        'placas_assignments': {}
    }

    # Guardar configuración
    with open(AB_TEST_CONFIG_PATH, 'w') as f:
        json.dump(ab_config, f, indent=2)

    print(f"✅ A/B Test configurado:")
    print(f"   📅 Duración: {duration_days} días")
    print(f"   🎯 Control: 60/40 (50% tráfico)")
    print(f"   📈 Treatment: 80/20 (50% tráfico)")
    print(f"   📊 Métricas: {len(ab_config['metrics']['primary'])} primarias, {len(ab_config['metrics']['secondary'])} secundarias")

    return ab_config


def assign_traffic(placas: List[str] = None) -> Dict[str, str]:
    """Asigna placas a variantes de A/B test."""

    print(f"\n🎯 ASIGNANDO TRÁFICO A VARIANTES")

    if not AB_TEST_CONFIG_PATH.exists():
        print("❌ A/B Test no configurado. Ejecuta --setup primero")
        return {}

    with open(AB_TEST_CONFIG_PATH, 'r') as f:
        ab_config = json.load(f)

    # Si no se proporcionan placas, cargar de datos existentes
    if placas is None:
        pia_data_path = ROOT / "data/processed/pia/pia_features_enhanced.csv"
        if pia_data_path.exists():
            pia_df = pd.read_csv(pia_data_path)
            placas = pia_df['placa'].unique().tolist()
        else:
            placas = ['A-05501-A', 'A-05355-A', 'A-05502-A', 'A-05503-A', 'A-05504-A', 'A-05507-A', 'A-05508-A']

    print(f"   📊 Placas a asignar: {len(placas)}")

    # Asignación determinística pero balanceada
    np.random.seed(42)  # Para reproducibilidad
    assignments = {}

    for i, placa in enumerate(placas):
        # Asignación 50/50 determinística
        variant = 'control' if i % 2 == 0 else 'treatment'
        assignments[placa] = variant

    # Actualizar configuración
    ab_config['placas_assignments'] = assignments
    ab_config['assignment_date'] = datetime.now().isoformat()

    with open(AB_TEST_CONFIG_PATH, 'w') as f:
        json.dump(ab_config, f, indent=2)

    control_count = sum(1 for v in assignments.values() if v == 'control')
    treatment_count = sum(1 for v in assignments.values() if v == 'treatment')

    print(f"   ✅ Asignación completada:")
    print(f"      Control (60/40): {control_count} placas")
    print(f"      Treatment (80/20): {treatment_count} placas")

    return assignments


def get_variant_for_placa(placa: str) -> str:
    """Obtiene la variante asignada para una placa específica."""

    if not AB_TEST_CONFIG_PATH.exists():
        return 'control'  # Default a control si no hay A/B test

    with open(AB_TEST_CONFIG_PATH, 'r') as f:
        ab_config = json.load(f)

    return ab_config.get('placas_assignments', {}).get(placa, 'control')


def apply_variant_weights(placa: str, core_risk: float, telemetry_risk: float) -> float:
    """Aplica pesos según variante asignada a la placa."""

    variant = get_variant_for_placa(placa)

    if variant == 'treatment':
        # Pesos optimizados 80/20
        return core_risk * 0.8 + telemetry_risk * 0.2
    else:
        # Pesos control 60/40
        return core_risk * 0.6 + telemetry_risk * 0.4


def collect_ab_metrics(results_df: pd.DataFrame) -> Dict[str, Any]:
    """Recolecta métricas del A/B test."""

    print(f"\n📊 RECOLECTANDO MÉTRICAS DE A/B TEST")

    if 'placa' not in results_df.columns:
        print("❌ DataFrame debe contener columna 'placa'")
        return {}

    # Agregar variante a cada registro
    results_df['variant'] = results_df['placa'].apply(get_variant_for_placa)

    control_data = results_df[results_df['variant'] == 'control']
    treatment_data = results_df[results_df['variant'] == 'treatment']

    if len(control_data) == 0 or len(treatment_data) == 0:
        print("⚠️ Una de las variantes no tiene datos")
        return {}

    metrics = {
        'timestamp': datetime.now().isoformat(),
        'sample_sizes': {
            'control': len(control_data),
            'treatment': len(treatment_data)
        },
        'metrics': {}
    }

    # Métricas primarias
    for variant_name, variant_data in [('control', control_data), ('treatment', treatment_data)]:
        metrics['metrics'][variant_name] = {}

        if 'overall_portfolio_risk' in variant_data.columns:
            risk_scores = variant_data['overall_portfolio_risk']
            metrics['metrics'][variant_name]['risk_score_mean'] = float(risk_scores.mean())
            metrics['metrics'][variant_name]['risk_score_std'] = float(risk_scores.std())
            metrics['metrics'][variant_name]['risk_score_range'] = float(risk_scores.max() - risk_scores.min())

        if 'risk_category' in variant_data.columns:
            risk_dist = variant_data['risk_category'].value_counts(normalize=True)
            metrics['metrics'][variant_name]['risk_distribution'] = risk_dist.to_dict()

        if 'projected_insurance_cost' in variant_data.columns:
            costs = variant_data['projected_insurance_cost']
            metrics['metrics'][variant_name]['avg_projected_cost'] = float(costs.mean())

    # Calcular diferencias
    if 'control' in metrics['metrics'] and 'treatment' in metrics['metrics']:
        control_range = metrics['metrics']['control'].get('risk_score_range', 0)
        treatment_range = metrics['metrics']['treatment'].get('risk_score_range', 0)

        if control_range > 0:
            discrimination_improvement = ((treatment_range - control_range) / control_range) * 100
            metrics['discrimination_improvement_pct'] = discrimination_improvement

        control_cost = metrics['metrics']['control'].get('avg_projected_cost', 0)
        treatment_cost = metrics['metrics']['treatment'].get('avg_projected_cost', 0)

        if control_cost > 0:
            cost_change_pct = ((treatment_cost - control_cost) / control_cost) * 100
            metrics['cost_change_pct'] = cost_change_pct

    return metrics


def analyze_ab_results() -> Dict[str, Any]:
    """Analiza resultados del A/B test y determina ganador."""

    print(f"\n🎯 ANALIZANDO RESULTADOS DE A/B TEST")

    # Cargar datos recientes
    pia_data_path = ROOT / "data/processed/pia/pia_features_enhanced.csv"
    if not pia_data_path.exists():
        print("❌ No hay datos PIA para analizar")
        return {}

    results_df = pd.read_csv(pia_data_path)
    metrics = collect_ab_metrics(results_df)

    if not metrics:
        return {}

    # Análisis de resultados
    analysis = {
        'test_date': datetime.now().isoformat(),
        'sample_sizes': metrics['sample_sizes'],
        'results_summary': {},
        'statistical_significance': {},
        'recommendation': {}
    }

    # Determinar ganador basado en discriminación
    discrimination_improvement = metrics.get('discrimination_improvement_pct', 0)
    cost_change = metrics.get('cost_change_pct', 0)

    print(f"   📊 Tamaños de muestra:")
    print(f"      Control: {metrics['sample_sizes']['control']}")
    print(f"      Treatment: {metrics['sample_sizes']['treatment']}")

    print(f"\n   📈 Resultados clave:")
    print(f"      Mejora en discriminación: {discrimination_improvement:+.1f}%")
    print(f"      Cambio en costos: {cost_change:+.1f}%")

    # Determinar recomendación
    if discrimination_improvement > 15:  # >15% mejora
        recommendation = "IMPLEMENTAR TREATMENT (80/20)"
        confidence = "ALTO"
        reason = f"Mejora significativa en discriminación (+{discrimination_improvement:.1f}%)"
    elif discrimination_improvement > 5:  # 5-15% mejora
        recommendation = "CONSIDERAR TREATMENT (80/20)"
        confidence = "MEDIO"
        reason = f"Mejora moderada en discriminación (+{discrimination_improvement:.1f}%)"
    elif discrimination_improvement > -5:  # -5% a 5%
        recommendation = "MANTENER CONTROL (60/40)"
        confidence = "MEDIO"
        reason = "No hay diferencia significativa"
    else:  # <-5%
        recommendation = "MANTENER CONTROL (60/40)"
        confidence = "ALTO"
        reason = f"Treatment muestra peor performance ({discrimination_improvement:+.1f}%)"

    analysis['recommendation'] = {
        'action': recommendation,
        'confidence': confidence,
        'reason': reason,
        'discrimination_improvement': discrimination_improvement,
        'cost_impact': cost_change
    }

    print(f"\n   🎯 RECOMENDACIÓN: {recommendation}")
    print(f"      Confianza: {confidence}")
    print(f"      Razón: {reason}")

    # Guardar resultados
    results_file = AB_TEST_RESULTS_PATH / f"ab_test_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(analysis, f, indent=2)

    print(f"\n   💾 Resultados guardados en: {results_file}")

    return analysis


def main() -> int:
    parser = argparse.ArgumentParser(description="Framework de A/B Testing para optimización de pesos")
    parser.add_argument('--setup', action='store_true',
                       help='Configurar nuevo A/B test')
    parser.add_argument('--assign-traffic', action='store_true',
                       help='Asignar placas a variantes')
    parser.add_argument('--analyze-results', action='store_true',
                       help='Analizar resultados y determinar ganador')
    parser.add_argument('--duration', type=int, default=14,
                       help='Duración del test en días (default: 14)')
    parser.add_argument('--test-name', type=str, default='pia_weights_optimization',
                       help='Nombre del A/B test')

    args = parser.parse_args()

    print("🧪 A/B TEST FRAMEWORK - OPTIMIZACIÓN DE PESOS PIA")
    print("=" * 55)

    try:
        if args.setup:
            setup_ab_test(args.test_name, args.duration)
            return 0

        elif args.assign_traffic:
            assignments = assign_traffic()
            if assignments:
                print(f"✅ {len(assignments)} placas asignadas a variantes")
            return 0

        elif args.analyze_results:
            analysis = analyze_ab_results()
            if analysis:
                recommendation = analysis['recommendation']['action']
                print(f"\n🎯 RESULTADO FINAL: {recommendation}")
                return 0 if 'IMPLEMENTAR' in recommendation else 1
            return 1

        else:
            print("❌ Especifica una acción: --setup, --assign-traffic, o --analyze-results")
            return 1

    except Exception as e:
        print(f"❌ Error en A/B test: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())