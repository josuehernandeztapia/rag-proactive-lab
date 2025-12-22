#!/usr/bin/env python3
"""Validación end-to-end de calidad del enriquecimiento completo.

Ejecuta todas las validaciones del framework de enriquecimiento:
- Validación matemática de pesos híbridos
- Validación temporal y estabilidad
- A/B testing framework
- Validación de datos outputs

Usage:
    python scripts/validation/validate_enrichment_quality.py --full-pipeline
    python scripts/validation/validate_enrichment_quality.py --quick-check
    python scripts/validation/validate_enrichment_quality.py --agents hase,pia
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any
import subprocess
import time
from datetime import datetime
import json

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Colores para output
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'  # No Color

def log(msg: str, color: str = Colors.NC) -> None:
    """Log con color y timestamp."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"{color}[{timestamp}] {msg}{Colors.NC}")

def run_validation_script(script_path: Path, args: List[str] = None) -> Dict[str, Any]:
    """Ejecuta un script de validación y captura resultado."""
    if not script_path.exists():
        return {
            'success': False,
            'error': f"Script no encontrado: {script_path}",
            'duration': 0
        }

    cmd = ["python", str(script_path)]
    if args:
        cmd.extend(args)

    start_time = time.time()

    try:
        log(f"🔄 Ejecutando: {script_path.name} {' '.join(args or [])}", Colors.BLUE)

        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=300  # 5 minutos max
        )

        duration = time.time() - start_time

        return {
            'success': result.returncode == 0,
            'returncode': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'duration': duration
        }

    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'error': 'Timeout (5 min)',
            'duration': time.time() - start_time
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'duration': time.time() - start_time
        }

def validate_data_quality() -> Dict[str, Any]:
    """Validación básica de calidad de datos."""
    log("🔍 Validando calidad de datos...", Colors.YELLOW)

    checks = {}

    # Verificar archivos clave existen
    key_files = [
        PROJECT_ROOT / 'data/processed/hase/enhanced_default_predictions.csv',
        PROJECT_ROOT / 'data/processed/pia/pia_features_enhanced.csv',
        PROJECT_ROOT / 'data/processed/guardian/guardian_insights.csv',
        PROJECT_ROOT / 'data/processed/pia/pia_hotspots.csv'
    ]

    for file_path in key_files:
        checks[file_path.name] = {
            'exists': file_path.exists(),
            'size_mb': file_path.stat().st_size / (1024 * 1024) if file_path.exists() else 0
        }

    # Verificar que archivos no estén vacíos
    non_empty = all(
        check['exists'] and check['size_mb'] > 0
        for check in checks.values()
    )

    return {
        'success': non_empty,
        'file_checks': checks,
        'total_files': len(key_files),
        'valid_files': sum(1 for c in checks.values() if c['exists'] and c['size_mb'] > 0)
    }

def validate_hybrid_weights(agents: List[str]) -> Dict[str, Any]:
    """Ejecuta validación de pesos híbridos para cada agente."""
    log("🧮 Validando pesos híbridos...", Colors.YELLOW)

    script_path = PROJECT_ROOT / 'scripts/validation/validate_hybrid_weights_simple.py'

    # Ejecutar validación para cada agente (el script original solo acepta uno a la vez)
    all_results = []
    overall_success = True

    for agent in agents:
        if agent.lower() in ['hase', 'pia']:  # Guardian no tiene validación de pesos
            log(f"  🔄 Validando {agent.upper()}...", Colors.BLUE)
            result = run_validation_script(script_path, ['--agent', agent.lower()])
            all_results.append({
                'agent': agent,
                'result': result
            })

            if not result['success']:
                overall_success = False
                log(f"  ❌ {agent.upper()}: {result.get('error', 'Unknown error')}", Colors.RED)
            else:
                log(f"  ✅ {agent.upper()}: Validación exitosa", Colors.GREEN)

    final_result = {
        'success': overall_success,
        'agent_results': all_results,
        'total_agents': len([a for a in agents if a.lower() in ['hase', 'pia']]),
        'passed_agents': sum(1 for r in all_results if r['result']['success'])
    }

    if overall_success:
        log("✅ Validación de pesos híbridos exitosa para todos los agentes", Colors.GREEN)
    else:
        failed = [r['agent'] for r in all_results if not r['result']['success']]
        log(f"❌ Validación de pesos híbridos falló para: {', '.join(failed)}", Colors.RED)

    return final_result

def validate_temporal_stability() -> Dict[str, Any]:
    """Ejecuta validación temporal y drift detection."""
    log("📈 Validando estabilidad temporal...", Colors.YELLOW)

    script_path = PROJECT_ROOT / 'scripts/validation/validate_temporal_stability.py'
    result = run_validation_script(script_path)

    if result['success']:
        log("✅ Validación temporal exitosa", Colors.GREEN)
    else:
        log(f"❌ Validación temporal falló: {result.get('error', 'Unknown error')}", Colors.RED)

    return result

def validate_ab_framework() -> Dict[str, Any]:
    """Validación del framework A/B testing."""
    log("🧪 Validando framework A/B...", Colors.YELLOW)

    script_path = PROJECT_ROOT / 'scripts/validation/ab_test_framework.py'
    result = run_validation_script(script_path)

    if result['success']:
        log("✅ Framework A/B validado exitosamente", Colors.GREEN)
    else:
        log(f"❌ Framework A/B falló: {result.get('error', 'Unknown error')}", Colors.RED)

    return result

def generate_validation_report(results: Dict[str, Any]) -> None:
    """Genera reporte de validación."""
    log("📋 Generando reporte de validación...", Colors.BLUE)

    # Crear directorio de reports
    reports_dir = PROJECT_ROOT / 'data/validation/quality_reports'
    reports_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = reports_dir / f'enrichment_quality_report_{timestamp}.json'

    # Calcular estadísticas summary
    total_tests = len([k for k in results.keys() if k != 'metadata'])
    passed_tests = sum(1 for k, v in results.items()
                      if k != 'metadata' and isinstance(v, dict) and v.get('success', False))

    results['metadata'] = {
        'report_date': datetime.now().isoformat(),
        'total_tests': total_tests,
        'passed_tests': passed_tests,
        'success_rate': passed_tests / total_tests if total_tests > 0 else 0
    }

    # Guardar reporte
    with open(report_file, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    log(f"📄 Reporte guardado: {report_file.relative_to(PROJECT_ROOT)}", Colors.GREEN)

    # Mostrar summary en consola
    print(f"\n{Colors.BLUE}📊 RESUMEN DE VALIDACIÓN:{Colors.NC}")
    print(f"   Tests ejecutados: {total_tests}")
    print(f"   Tests exitosos: {passed_tests}")
    print(f"   Tasa de éxito: {passed_tests/total_tests*100:.1f}%")

    if passed_tests == total_tests:
        print(f"{Colors.GREEN}✅ TODAS LAS VALIDACIONES PASARON{Colors.NC}")
    else:
        print(f"{Colors.RED}❌ {total_tests - passed_tests} VALIDACIONES FALLARON{Colors.NC}")

def main() -> int:
    parser = argparse.ArgumentParser(description='Validación end-to-end de calidad del enriquecimiento')

    parser.add_argument('--full-pipeline', action='store_true',
                       help='Ejecutar validación completa del pipeline')
    parser.add_argument('--quick-check', action='store_true',
                       help='Validación rápida (solo calidad de datos)')
    parser.add_argument('--agents', default='hase,pia,guardian',
                       help='Agentes a validar (separados por coma)')
    parser.add_argument('--skip-temporal', action='store_true',
                       help='Omitir validación temporal (más rápido)')
    parser.add_argument('--skip-ab', action='store_true',
                       help='Omitir validación A/B (más rápido)')

    args = parser.parse_args()

    agents = [a.strip() for a in args.agents.split(',')]

    print(f"\n{Colors.BLUE}🔬 VALIDACIÓN END-TO-END DE CALIDAD{Colors.NC}")
    print(f"{Colors.BLUE}===================================={Colors.NC}")
    print(f"📅 Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎯 Agentes: {', '.join(agents)}")
    print()

    results = {}

    # 1. Validación básica de calidad de datos (siempre)
    results['data_quality'] = validate_data_quality()

    # 2. Validación rápida (solo datos)
    if args.quick_check:
        log("⚡ Modo quick-check: solo validación de datos", Colors.YELLOW)
        generate_validation_report(results)
        return 0 if results['data_quality']['success'] else 1

    # 3. Validación de pesos híbridos
    results['hybrid_weights'] = validate_hybrid_weights(agents)

    # 4. Validación temporal (opcional)
    if not args.skip_temporal:
        results['temporal_stability'] = validate_temporal_stability()
    else:
        log("⏭️ Omitiendo validación temporal", Colors.YELLOW)

    # 5. Validación A/B framework (opcional)
    if not args.skip_ab and args.full_pipeline:
        results['ab_framework'] = validate_ab_framework()
    else:
        log("⏭️ Omitiendo validación A/B", Colors.YELLOW)

    # 6. Generar reporte final
    generate_validation_report(results)

    # Determinar resultado final
    failed_tests = [k for k, v in results.items()
                   if k != 'metadata' and isinstance(v, dict) and not v.get('success', False)]

    if failed_tests:
        log(f"❌ Validación falló en: {', '.join(failed_tests)}", Colors.RED)
        return 1
    else:
        log("✅ Todas las validaciones pasaron exitosamente", Colors.GREEN)
        return 0

if __name__ == "__main__":
    sys.exit(main())