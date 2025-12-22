#!/usr/bin/env python3
"""Validación matemática simplificada de pesos híbridos para agentes enriquecidos.

Usage:
    python scripts/validation/validate_hybrid_weights_simple.py
    python scripts/validation/validate_hybrid_weights_simple.py --agent hase
    python scripts/validation/validate_hybrid_weights_simple.py --optimize-weights

Solo usa pandas y numpy - no requiere sklearn.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_agent_data(agent: str) -> pd.DataFrame:
    """Carga datos enriquecidos de un agente específico."""

    data_paths = {
        'hase': ROOT / "data/processed/hase/enhanced_default_predictions.csv",
        'pia': ROOT / "data/processed/pia/pia_features_enhanced.csv",
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
    print(f"📊 Dataset shape: {df.shape}")
    print(f"📋 Columnas disponibles: {list(df.columns)}")

    if agent == 'hase':
        # HASE: Default prediction system
        core_feature = 'core_default_risk'
        telemetry_feature = 'behavioral_default_risk'
        target = 'enhanced_default_risk'
    elif agent == 'pia':
        # PIA: Portfolio risk system
        core_feature = 'core_financial_risk'
        telemetry_feature = 'telemetry_enhancement_score'
        target = 'overall_portfolio_risk'
    else:
        print("❌ Agente no soportado")
        return {}

    # Verificar que las columnas existen
    required_cols = [core_feature, telemetry_feature, target]
    available_cols = [col for col in required_cols if col in df.columns]

    print(f"\n🔍 FEATURES REQUERIDOS: {required_cols}")
    print(f"✅ FEATURES DISPONIBLES: {available_cols}")

    if len(available_cols) < len(required_cols):
        missing = set(required_cols) - set(available_cols)
        print(f"❌ Columnas faltantes: {missing}")
        return {'error': f'Missing columns: {missing}'}

    # Limpiar datos (quitar NaN)
    clean_data = df[required_cols].dropna()
    print(f"📊 Registros válidos para análisis: {len(clean_data)}")

    if len(clean_data) < 5:
        print("❌ Datos insuficientes para análisis")
        return {'error': 'Insufficient data'}

    # Calcular correlaciones
    correlation_matrix = clean_data.corr()

    print(f"\n📊 MATRIZ DE CORRELACIONES:")
    print(correlation_matrix.round(4))

    # Análisis específico
    core_telemetry_corr = correlation_matrix.loc[core_feature, telemetry_feature]
    core_target_corr = correlation_matrix.loc[core_feature, target]
    telemetry_target_corr = correlation_matrix.loc[telemetry_feature, target]

    print(f"\n🎯 ANÁLISIS DE INDEPENDENCIA:")
    print(f"   📊 {core_feature} ↔ {telemetry_feature}: {core_telemetry_corr:.4f}")

    if abs(core_telemetry_corr) < 0.3:
        independence_status = "✅ ALTAMENTE INDEPENDIENTES - Excelente complementariedad"
        independence_score = "EXCELENTE"
    elif abs(core_telemetry_corr) < 0.6:
        independence_status = "⚠️ MODERADAMENTE CORRELACIONADOS - Aún útil"
        independence_score = "BUENO"
    else:
        independence_status = "❌ ALTAMENTE CORRELACIONADOS - Posible redundancia"
        independence_score = "PROBLEMÁTICO"

    print(f"   🏆 Status: {independence_status}")

    print(f"\n🎯 CONTRIBUCIÓN AL TARGET:")
    print(f"   📈 {core_feature} → {target}: {core_target_corr:.4f}")
    print(f"   📈 {telemetry_feature} → {target}: {telemetry_target_corr:.4f}")

    # Estadísticas descriptivas
    print(f"\n📊 ESTADÍSTICAS DESCRIPTIVAS:")
    for col in required_cols:
        data_col = clean_data[col]
        print(f"   {col}:")
        print(f"      Mean: {data_col.mean():.4f}, Std: {data_col.std():.4f}")
        print(f"      Min: {data_col.min():.4f}, Max: {data_col.max():.4f}")

    return {
        'correlation_matrix': correlation_matrix,
        'core_telemetry_correlation': core_telemetry_corr,
        'core_target_correlation': core_target_corr,
        'telemetry_target_correlation': telemetry_target_corr,
        'independence_status': independence_status,
        'independence_score': independence_score,
        'clean_data': clean_data,
        'features': {
            'core': core_feature,
            'telemetry': telemetry_feature,
            'target': target
        }
    }


def optimize_weights_grid_search(correlation_analysis: Dict[str, Any], agent: str) -> Dict[str, Any]:
    """Optimiza pesos usando grid search simple."""

    print(f"\n🔬 OPTIMIZACIÓN DE PESOS - {agent.upper()}")

    features = correlation_analysis.get('features', {})
    clean_data = correlation_analysis.get('clean_data')

    if clean_data is None or len(clean_data) < 5:
        print("❌ Datos insuficientes para optimización")
        return {'error': 'Insufficient data'}

    core_feature = features['core']
    telemetry_feature = features['telemetry']

    # Grid search de pesos
    weight_combinations = [
        (0.5, 0.5, "50/50 - Equilibrado"),
        (0.6, 0.4, "60/40 - PIA anterior"),
        (0.8, 0.2, "80/20 - PIA actual optimizado"),
        (0.65, 0.35, "65/35"),
        (0.7, 0.3, "70/30 - HASE actual"),
        (0.75, 0.25, "75/25"),
        (0.8, 0.2, "80/20 - Core dominante"),
        (0.85, 0.15, "85/15"),
        (0.9, 0.1, "90/10 - Casi solo core")
    ]

    results = []

    print(f"\n📊 PROBANDO {len(weight_combinations)} COMBINACIONES:")
    print("Pesos      | Híbrido Score |  Rango  |  Std   | Descripción")
    print("-" * 65)

    for core_weight, telemetry_weight, description in weight_combinations:
        # Calcular scoring híbrido
        hybrid_score = (
            clean_data[core_feature] * core_weight +
            clean_data[telemetry_feature] * telemetry_weight
        )

        # Métricas de calidad del score
        score_mean = hybrid_score.mean()
        score_std = hybrid_score.std()
        score_range = hybrid_score.max() - hybrid_score.min()

        results.append({
            'core_weight': core_weight,
            'telemetry_weight': telemetry_weight,
            'description': description,
            'hybrid_score_mean': score_mean,
            'hybrid_score_std': score_std,
            'hybrid_score_range': score_range,
            'hybrid_scores': hybrid_score.values
        })

        print(f"{core_weight:.1f}/{telemetry_weight:.1f} | {score_mean:11.4f} | {score_range:7.4f} | {score_std:6.4f} | {description}")

    # Encontrar mejor combinación (mayor rango para mejor discriminación)
    best_idx = np.argmax([r['hybrid_score_range'] for r in results])
    best_combination = results[best_idx]

    print(f"\n🎯 MEJOR COMBINACIÓN (mayor discriminación):")
    print(f"   Pesos: {best_combination['core_weight']:.1f}/{best_combination['telemetry_weight']:.1f}")
    print(f"   Descripción: {best_combination['description']}")
    print(f"   Score Range: {best_combination['hybrid_score_range']:.4f}")
    print(f"   Score Std: {best_combination['hybrid_score_std']:.4f}")

    return {
        'optimization_results': results,
        'best_weights': (best_combination['core_weight'], best_combination['telemetry_weight']),
        'best_combination': best_combination
    }


def validate_current_weights(correlation_analysis: Dict[str, Any], agent: str) -> Dict[str, Any]:
    """Valida los pesos actuales implementados."""

    print(f"\n✅ VALIDACIÓN DE PESOS ACTUALES - {agent.upper()}")

    # Pesos actuales implementados
    current_weights = {
        'hase': (0.7, 0.3),  # 70% core, 30% telemetría
        'pia': (0.8, 0.2),   # 80% core, 20% telemetría (OPTIMIZADO matemáticamente)
    }

    if agent not in current_weights:
        print(f"❌ Pesos no definidos para {agent}")
        return {'error': f'Weights not defined for {agent}'}

    current_core, current_telemetry = current_weights[agent]

    features = correlation_analysis.get('features', {})
    clean_data = correlation_analysis.get('clean_data')

    if clean_data is None:
        return {'error': 'No clean data available'}

    core_feature = features['core']
    telemetry_feature = features['telemetry']
    target_feature = features['target']

    # Calcular score con pesos actuales
    current_hybrid_score = (
        clean_data[core_feature] * current_core +
        clean_data[telemetry_feature] * current_telemetry
    )

    # Comparar con target real
    target_correlation = np.corrcoef(current_hybrid_score, clean_data[target_feature])[0, 1]

    print(f"   Pesos implementados: {current_core:.1f}/{current_telemetry:.1f} (core/telemetría)")
    print(f"   Correlación híbrido ↔ target: {target_correlation:.4f}")

    # Calcular métricas de calidad
    score_mean = current_hybrid_score.mean()
    score_std = current_hybrid_score.std()
    score_range = current_hybrid_score.max() - current_hybrid_score.min()

    print(f"   Score Mean: {score_mean:.4f}")
    print(f"   Score Std: {score_std:.4f}")
    print(f"   Score Range: {score_range:.4f}")

    if target_correlation > 0.95:
        quality_assessment = "✅ EXCELENTE - Pesos perfectamente calibrados"
    elif target_correlation > 0.9:
        quality_assessment = "✅ MUY BUENO - Pesos muy bien calibrados"
    elif target_correlation > 0.8:
        quality_assessment = "✅ BUENO - Pesos adecuados"
    elif target_correlation > 0.6:
        quality_assessment = "⚠️ MODERADO - Considerar optimización"
    else:
        quality_assessment = "❌ POBRE - Requiere optimización urgente"

    print(f"   🏆 Evaluación: {quality_assessment}")

    return {
        'current_weights': current_weights[agent],
        'target_correlation': target_correlation,
        'score_stats': {
            'mean': score_mean,
            'std': score_std,
            'range': score_range
        },
        'quality_assessment': quality_assessment
    }


def generate_synthetic_scenarios(agent: str, optimal_weights: Tuple[float, float]) -> Dict[str, Any]:
    """Genera escenarios sintéticos para validar pesos."""

    print(f"\n🧪 VALIDACIÓN CON ESCENARIOS SINTÉTICOS - {agent.upper()}")

    core_weight, telemetry_weight = optimal_weights

    scenarios = [
        ('Alto Core, Bajo Telemetría', 0.9, 0.1, 'Alto riesgo financiero, comportamiento normal'),
        ('Bajo Core, Alto Telemetría', 0.1, 0.9, 'Perfil financiero sano, comportamiento riesgoso'),
        ('Ambos Altos', 0.9, 0.9, 'Alto riesgo en ambos componentes - CRÍTICO'),
        ('Ambos Bajos', 0.1, 0.1, 'Bajo riesgo en ambos componentes - SEGURO'),
        ('Equilibrio Alto', 0.6, 0.6, 'Riesgo moderado-alto balanceado'),
        ('Equilibrio Bajo', 0.3, 0.3, 'Riesgo moderado-bajo balanceado')
    ]

    print(f"   Usando pesos: {core_weight:.1f}/{telemetry_weight:.1f} (core/telemetría)")
    print(f"\n📊 ESCENARIOS DE VALIDACIÓN:")
    print("Escenario                   | Core | Tele | Híbrido | Interpretación")
    print("-" * 85)

    scenario_results = []

    for name, core_risk, telemetry_risk, interpretation in scenarios:
        hybrid_score = core_weight * core_risk + telemetry_weight * telemetry_risk

        scenario_results.append({
            'name': name,
            'core_risk': core_risk,
            'telemetry_risk': telemetry_risk,
            'hybrid_score': hybrid_score,
            'interpretation': interpretation
        })

        print(f"{name:<25} | {core_risk:.1f}  | {telemetry_risk:.1f}  |  {hybrid_score:.3f}  | {interpretation}")

    # Análisis de rankings
    sorted_scenarios = sorted(scenario_results, key=lambda x: x['hybrid_score'])

    print(f"\n🎯 RANKING DE RIESGO (menor a mayor):")
    for i, scenario in enumerate(sorted_scenarios, 1):
        print(f"   {i}. {scenario['name']} (Score: {scenario['hybrid_score']:.3f})")

    # Validar separación adecuada
    scores = [s['hybrid_score'] for s in scenario_results]
    score_range = max(scores) - min(scores)

    print(f"\n📊 ANÁLISIS DE DISCRIMINACIÓN:")
    print(f"   Range total: {score_range:.3f}")
    print(f"   Separación: {'✅ Adecuada' if score_range > 0.5 else '⚠️ Limitada' if score_range > 0.3 else '❌ Insuficiente'}")

    return {
        'scenarios': scenario_results,
        'score_range': score_range,
        'weights_used': optimal_weights,
        'ranking': sorted_scenarios
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validación matemática simplificada de pesos híbridos")
    parser.add_argument('--agent', choices=['hase', 'pia'], default='all',
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
        print(f"\n{'='*15} ANALIZANDO {agent.upper()} {'='*15}")

        try:
            # 1. Cargar datos
            df = load_agent_data(agent)
            print(f"📊 Datos cargados: {len(df)} registros")

            # 2. Análisis de correlaciones
            correlation_analysis = analyze_feature_correlations(df, agent)

            if 'error' in correlation_analysis:
                print(f"❌ Error en análisis de correlaciones: {correlation_analysis['error']}")
                all_results[agent] = correlation_analysis
                continue

            # 3. Validación de pesos actuales
            current_validation = validate_current_weights(correlation_analysis, agent)

            # 4. Optimización de pesos (opcional)
            optimization_results = {}
            if args.optimize_weights:
                optimization_results = optimize_weights_grid_search(correlation_analysis, agent)

            # 5. Validación sintética (opcional)
            synthetic_validation = {}
            if args.synthetic_validation:
                if optimization_results and 'best_weights' in optimization_results:
                    optimal_weights = optimization_results['best_weights']
                else:
                    # Usar pesos actuales si no hay optimización
                    current_weights = {'hase': (0.7, 0.3), 'pia': (0.6, 0.4)}
                    optimal_weights = current_weights.get(agent, (0.7, 0.3))

                synthetic_validation = generate_synthetic_scenarios(agent, optimal_weights)

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

        if correlation and 'independence_score' in correlation:
            independence_score = correlation.get('independence_score', 'N/A')
            core_telemetry_corr = correlation.get('core_telemetry_correlation', 0)
            print(f"   📊 Independencia features: {independence_score} (r={core_telemetry_corr:.3f})")

        if validation and 'quality_assessment' in validation:
            current_weights = validation.get('current_weights', (0, 0))
            target_corr = validation.get('target_correlation', 0)
            print(f"   ⚖️  Pesos actuales: {current_weights[0]:.1f}/{current_weights[1]:.1f} (corr_target={target_corr:.3f})")

        if optimization and 'best_weights' in optimization:
            optimal_weights = optimization.get('best_weights', (0, 0))
            best_combination = optimization.get('best_combination', {})
            best_range = best_combination.get('hybrid_score_range', 0)
            print(f"   🎯 Pesos óptimos sugeridos: {optimal_weights[0]:.1f}/{optimal_weights[1]:.1f} (range={best_range:.3f})")

    print(f"\n🎉 Análisis completado para {len(all_results)} agentes")

    return 0


if __name__ == "__main__":
    sys.exit(main())