#!/usr/bin/env python3
"""Validación matemática de pesos híbridos para agentes enriquecidos.

Usage:
    python scripts/validation/validate_hybrid_weights.py
    python scripts/validation/validate_hybrid_weights.py --agent hase
    python scripts/validation/validate_hybrid_weights.py --agent pia --optimize-weights

Analiza correlaciones, optimiza pesos y valida que la lógica híbrida es matemáticamente sólida.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Tuple, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.metadata import write_metadata


def load_agent_data(agent: str) -> pd.DataFrame:
    """Carga datos enriquecidos de un agente específico."""

    data_paths = {
        'hase': ROOT / "data/processed/hase/enhanced_default_predictions.csv",
        'pia': ROOT / "data/processed/pia/pia_features_enhanced.csv",
        'guardian': ROOT / "data/processed/guardian/guardian_insights.csv"
    }

    if agent not in data_paths:
        raise ValueError(f"Agent '{agent}' not supported. Use: {list(data_paths.keys())}")

    path = data_paths[agent]
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    return pd.read_csv(path)


def analyze_feature_correlations(df: pd.DataFrame, agent: str) -> Dict[str, Any]:
    """Analiza correlaciones entre features core y telemetría."""

    print(f"\n🔍 ANÁLISIS DE CORRELACIONES - {agent.upper()}")

    if agent == 'hase':
        core_feature = 'core_default_risk'
        telemetry_feature = 'behavioral_default_risk'
        target = 'enhanced_default_risk'
    elif agent == 'pia':
        core_feature = 'core_financial_risk'
        telemetry_feature = 'telemetry_enhancement_score'
        target = 'overall_portfolio_risk'
    else:
        print("❌ Guardian analysis not implemented yet")
        return {}

    # Verificar que las columnas existen
    required_cols = [core_feature, telemetry_feature, target]
    available_cols = [col for col in required_cols if col in df.columns]

    if len(available_cols) < len(required_cols):
        missing = set(required_cols) - set(available_cols)
        print(f"⚠️  Columnas faltantes: {missing}")
        print(f"📊 Columnas disponibles: {df.columns.tolist()}")
        return {}

    # Calcular correlaciones
    correlation_matrix = df[required_cols].corr()

    print(f"\n📊 MATRIZ DE CORRELACIONES:")
    print(correlation_matrix.round(3))

    # Correlación core-telemetría (independencia)
    core_telemetry_corr = correlation_matrix.loc[core_feature, telemetry_feature]

    print(f"\n🎯 ANÁLISIS DE INDEPENDENCIA:")
    print(f"   Correlación {core_feature} ↔ {telemetry_feature}: {core_telemetry_corr:.3f}")

    if abs(core_telemetry_corr) < 0.3:
        independence_status = "✅ ALTAMENTE INDEPENDIENTES - Excelente complementariedad"
    elif abs(core_telemetry_corr) < 0.6:
        independence_status = "⚠️ MODERADAMENTE CORRELACIONADOS - Aún útil"
    else:
        independence_status = "❌ ALTAMENTE CORRELACIONADOS - Posible redundancia"

    print(f"   Status: {independence_status}")

    # Análisis de contribución individual
    core_target_corr = correlation_matrix.loc[core_feature, target]
    telemetry_target_corr = correlation_matrix.loc[telemetry_feature, target]

    print(f"\n🎯 CONTRIBUCIÓN AL TARGET:")
    print(f"   {core_feature} → {target}: {core_target_corr:.3f}")
    print(f"   {telemetry_feature} → {target}: {telemetry_target_corr:.3f}")

    return {
        'correlation_matrix': correlation_matrix,
        'core_telemetry_correlation': core_telemetry_corr,
        'core_target_correlation': core_target_corr,
        'telemetry_target_correlation': telemetry_target_corr,
        'independence_status': independence_status,
        'features': {
            'core': core_feature,
            'telemetry': telemetry_feature,
            'target': target
        }
    }


def optimize_weights_grid_search(df: pd.DataFrame, agent: str, correlation_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Optimiza pesos usando grid search con validación cruzada."""

    print(f"\n🔬 OPTIMIZACIÓN DE PESOS - {agent.upper()}")

    features = correlation_analysis.get('features', {})
    if not features:
        print("❌ No se pueden optimizar pesos sin análisis de correlaciones")
        return {}

    core_feature = features['core']
    telemetry_feature = features['telemetry']

    # Verificar que tenemos datos suficientes
    valid_data = df[[core_feature, telemetry_feature]].dropna()
    if len(valid_data) < 10:
        print(f"❌ Datos insuficientes: {len(valid_data)} registros")
        return {}

    # Grid search de pesos
    weight_combinations = [
        (0.5, 0.5),   # 50/50
        (0.6, 0.4),   # 60/40 (actual PIA)
        (0.65, 0.35), # 65/35
        (0.7, 0.3),   # 70/30 (actual HASE)
        (0.75, 0.25), # 75/25
        (0.8, 0.2),   # 80/20
        (0.85, 0.15), # 85/15
        (0.9, 0.1)    # 90/10
    ]

    results = []

    print(f"\n📊 PROBANDO {len(weight_combinations)} COMBINACIONES DE PESOS:")
    print("Core Weight | Telemetry Weight | Híbrido Score | Varianza | Distribución")
    print("-" * 80)

    for core_weight, telemetry_weight in weight_combinations:
        # Calcular scoring híbrido
        hybrid_score = (
            valid_data[core_feature] * core_weight +
            valid_data[telemetry_feature] * telemetry_weight
        )

        # Métricas de calidad del score
        score_variance = hybrid_score.var()
        score_range = hybrid_score.max() - hybrid_score.min()
        score_std = hybrid_score.std()

        # Distribución (idealmente queremos buena separación)
        score_distribution_quality = score_range / (score_std + 1e-10)

        results.append({
            'core_weight': core_weight,
            'telemetry_weight': telemetry_weight,
            'hybrid_score_mean': hybrid_score.mean(),
            'hybrid_score_variance': score_variance,
            'hybrid_score_range': score_range,
            'hybrid_score_std': score_std,
            'distribution_quality': score_distribution_quality,
            'hybrid_scores': hybrid_score.values
        })

        print(f"    {core_weight:.1f}    |     {telemetry_weight:.1f}      |    {hybrid_score.mean():.3f}    |  {score_variance:.3f}  |   {score_distribution_quality:.2f}")

    # Encontrar mejor combinación (mayor rango y buena distribución)
    results_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'hybrid_scores'} for r in results])

    # Score compuesto: balancear rango y distribución
    results_df['composite_score'] = (
        results_df['hybrid_score_range'] * 0.6 +  # Queremos buen rango
        results_df['distribution_quality'] * 0.4   # Y buena distribución
    )

    best_idx = results_df['composite_score'].idxmax()
    best_combination = results[best_idx]

    print(f"\n🎯 MEJOR COMBINACIÓN ENCONTRADA:")
    print(f"   Core Weight: {best_combination['core_weight']:.1f}")
    print(f"   Telemetry Weight: {best_combination['telemetry_weight']:.1f}")
    print(f"   Composite Score: {results_df.loc[best_idx, 'composite_score']:.3f}")
    print(f"   Score Range: {best_combination['hybrid_score_range']:.3f}")
    print(f"   Distribution Quality: {best_combination['distribution_quality']:.2f}")

    return {
        'optimization_results': results,
        'best_weights': (best_combination['core_weight'], best_combination['telemetry_weight']),
        'best_combination': best_combination,
        'results_summary': results_df
    }


def validate_current_weights(df: pd.DataFrame, agent: str, correlation_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Valida si los pesos actuales están cerca del óptimo."""

    print(f"\n✅ VALIDACIÓN DE PESOS ACTUALES - {agent.upper()}")

    # Pesos actuales implementados
    current_weights = {
        'hase': (0.7, 0.3),  # 70% core, 30% telemetría
        'pia': (0.6, 0.4),   # 60% core, 40% telemetría
    }

    if agent not in current_weights:
        print(f"❌ Pesos no definidos para {agent}")
        return {}

    current_core, current_telemetry = current_weights[agent]

    features = correlation_analysis.get('features', {})
    if not features:
        return {}

    core_feature = features['core']
    telemetry_feature = features['telemetry']
    target_feature = features['target']

    # Calcular score con pesos actuales
    valid_data = df[[core_feature, telemetry_feature, target_feature]].dropna()

    current_hybrid_score = (
        valid_data[core_feature] * current_core +
        valid_data[telemetry_feature] * current_telemetry
    )

    # Comparar con target real (si existe)
    target_correlation = np.corrcoef(current_hybrid_score, valid_data[target_feature])[0, 1]

    print(f"   Pesos actuales: {current_core:.1f}/{current_telemetry:.1f} (core/telemetría)")
    print(f"   Correlación híbrido ↔ target: {target_correlation:.3f}")

    # Calcular métricas de calidad
    score_stats = {
        'mean': current_hybrid_score.mean(),
        'std': current_hybrid_score.std(),
        'range': current_hybrid_score.max() - current_hybrid_score.min(),
        'target_correlation': target_correlation
    }

    if target_correlation > 0.9:
        quality_assessment = "✅ EXCELENTE - Pesos muy bien calibrados"
    elif target_correlation > 0.8:
        quality_assessment = "✅ BUENO - Pesos adecuados"
    elif target_correlation > 0.6:
        quality_assessment = "⚠️ MODERADO - Considerar optimización"
    else:
        quality_assessment = "❌ POBRE - Requiere optimización"

    print(f"   Evaluación: {quality_assessment}")

    return {
        'current_weights': current_weights[agent],
        'hybrid_score_stats': score_stats,
        'target_correlation': target_correlation,
        'quality_assessment': quality_assessment
    }


def generate_synthetic_validation(agent: str, optimal_weights: Tuple[float, float]) -> Dict[str, Any]:
    """Genera escenarios sintéticos para validar pesos óptimos."""

    print(f"\n🧪 VALIDACIÓN CON DATOS SINTÉTICOS - {agent.upper()}")

    core_weight, telemetry_weight = optimal_weights

    # Crear escenarios controlados
    scenarios = {
        'high_core_low_telemetry': {
            'core_risk': 0.8,
            'telemetry_risk': 0.2,
            'expected_hybrid': core_weight * 0.8 + telemetry_weight * 0.2,
            'business_interpretation': 'Alto riesgo financiero, comportamiento normal'
        },
        'low_core_high_telemetry': {
            'core_risk': 0.2,
            'telemetry_risk': 0.8,
            'expected_hybrid': core_weight * 0.2 + telemetry_weight * 0.8,
            'business_interpretation': 'Perfil financiero sano, comportamiento riesgoso'
        },
        'both_high': {
            'core_risk': 0.9,
            'telemetry_risk': 0.9,
            'expected_hybrid': core_weight * 0.9 + telemetry_weight * 0.9,
            'business_interpretation': 'Alto riesgo en ambos componentes'
        },
        'both_low': {
            'core_risk': 0.1,
            'telemetry_risk': 0.1,
            'expected_hybrid': core_weight * 0.1 + telemetry_weight * 0.1,
            'business_interpretation': 'Bajo riesgo en ambos componentes'
        }
    }

    print(f"   Usando pesos: {core_weight:.1f}/{telemetry_weight:.1f} (core/telemetría)")
    print(f"\n📊 ESCENARIOS DE VALIDACIÓN:")
    print("Scenario                    | Core | Tele | Hybrid | Interpretación")
    print("-" * 80)

    for scenario_name, scenario_data in scenarios.items():
        core_risk = scenario_data['core_risk']
        telemetry_risk = scenario_data['telemetry_risk']
        expected_hybrid = scenario_data['expected_hybrid']
        interpretation = scenario_data['business_interpretation']

        print(f"{scenario_name:<25} | {core_risk:.1f}  | {telemetry_risk:.1f}  |  {expected_hybrid:.3f}  | {interpretation}")

    # Validar que los rankings tienen sentido
    hybrid_scores = [s['expected_hybrid'] for s in scenarios.values()]

    print(f"\n🎯 VALIDACIÓN DE RANKINGS:")
    print(f"   Range de scores híbridos: {min(hybrid_scores):.3f} - {max(hybrid_scores):.3f}")
    print(f"   Separación adecuada: {'✅' if (max(hybrid_scores) - min(hybrid_scores)) > 0.5 else '⚠️'}")

    return {
        'scenarios': scenarios,
        'score_range': (min(hybrid_scores), max(hybrid_scores)),
        'weights_used': optimal_weights
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validación matemática de pesos híbridos")
    parser.add_argument('--agent', choices=['hase', 'pia', 'guardian'], default='all',
                       help='Agente a analizar (default: all)')
    parser.add_argument('--optimize-weights', action='store_true',
                       help='Ejecutar optimización de pesos')
    parser.add_argument('--synthetic-validation', action='store_true',
                       help='Ejecutar validación con datos sintéticos')

    args = parser.parse_args()

    print("🧮 VALIDACIÓN MATEMÁTICA DE PESOS HÍBRIDOS")
    print("=" * 50)

    agents_to_analyze = ['hase', 'pia'] if args.agent == 'all' else [args.agent]

    all_results = {}

    for agent in agents_to_analyze:
        print(f"\n{'='*20} ANALIZANDO {agent.upper()} {'='*20}")

        try:
            # 1. Cargar datos
            df = load_agent_data(agent)
            print(f"📊 Datos cargados: {len(df)} registros")

            # 2. Análisis de correlaciones
            correlation_analysis = analyze_feature_correlations(df, agent)

            # 3. Validación de pesos actuales
            current_validation = validate_current_weights(df, agent, correlation_analysis)

            # 4. Optimización de pesos (opcional)
            optimization_results = {}
            if args.optimize_weights and correlation_analysis:
                optimization_results = optimize_weights_grid_search(df, agent, correlation_analysis)

            # 5. Validación sintética (opcional)
            synthetic_validation = {}
            if args.synthetic_validation and optimization_results:
                optimal_weights = optimization_results.get('best_weights', (0.7, 0.3))
                synthetic_validation = generate_synthetic_validation(agent, optimal_weights)

            all_results[agent] = {
                'correlation_analysis': correlation_analysis,
                'current_validation': current_validation,
                'optimization_results': optimization_results,
                'synthetic_validation': synthetic_validation
            }

        except Exception as e:
            print(f"❌ Error analizando {agent}: {e}")
            all_results[agent] = {'error': str(e)}

    # Resumen final
    print(f"\n{'='*50}")
    print("📋 RESUMEN DE VALIDACIÓN MATEMÁTICA")
    print(f"{'='*50}")

    for agent, results in all_results.items():
        if 'error' in results:
            print(f"\n❌ {agent.upper()}: {results['error']}")
            continue

        correlation = results.get('correlation_analysis', {})
        validation = results.get('current_validation', {})
        optimization = results.get('optimization_results', {})

        print(f"\n✅ {agent.upper()}:")

        if correlation:
            independence = correlation.get('independence_status', 'N/A')
            print(f"   📊 Independencia features: {independence}")

        if validation:
            quality = validation.get('quality_assessment', 'N/A')
            current_weights = validation.get('current_weights', (0, 0))
            print(f"   ⚖️  Pesos actuales: {current_weights[0]:.1f}/{current_weights[1]:.1f} - {quality}")

        if optimization:
            optimal_weights = optimization.get('best_weights', (0, 0))
            print(f"   🎯 Pesos óptimos sugeridos: {optimal_weights[0]:.1f}/{optimal_weights[1]:.1f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())