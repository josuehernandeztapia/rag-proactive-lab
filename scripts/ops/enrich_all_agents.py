#!/usr/bin/env python3
"""Proceso completo de enriquecimiento de telemetría para todos los agentes.

Este script orquesta el pipeline completo de:
1. Extracción enriquecida desde Geotab
2. Procesamiento inteligente de eventos
3. Distribución a todos los agentes
4. Generación de insights enriquecidos

Usage:
    python scripts/ops/enrich_all_agents.py
    python scripts/ops/enrich_all_agents.py --date 2025-12-21 --force-reprocess
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List

# Configuración de componentes del ecosistema
COMPONENTS = {
    'guardian': {
        'script': 'agents/guardian/scripts/build_insights.py',
        'config': 'config/guardian.yml',
        'dependencies': ['events_daily', 'events_percentiles'],
        'status': '✅ IMPLEMENTADO'
    },
    'pia': {
        'script': 'agents/pia/scripts/build_enhanced_dataset.py',
        'config': 'config/pia.yml',
        'dependencies': ['events_daily', 'trips_daily'],
        'status': '🔄 ENRIQUECIDO (con eventos de riesgo)'
    },
    'hase': {
        'script': 'agents/hase/scripts/build_enhanced_features.py',
        'config': 'config/hase.yml',
        'dependencies': ['events_daily', 'trips_daily'],
        'status': '🔄 ENRIQUECIDO (sin GNV)'
    },
    'dashboards': {
        'script': 'agents/pia/scripts/build_pia_dashboard_hotspots.py',
        'config': None,
        'dependencies': ['events_daily', 'pia_features'],
        'status': '🔄 PENDIENTE'
    }
}

def run_command(cmd: List[str], description: str) -> bool:
    """Ejecuta un comando y maneja errores."""
    print(f"🔄 {description}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ {description} - OK")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - FAILED")
        print(f"Error: {e.stderr}")
        return False

def check_dependencies(component: str, deps: List[str], date_suffix: str) -> bool:
    """Verifica que las dependencias existan."""
    missing = []

    for dep in deps:
        if dep == 'events_daily':
            path = Path(f"data/staging/geotab_events_daily_{date_suffix}-refined.csv")
        elif dep == 'events_percentiles':
            path = Path(f"data/staging/geotab_events_market_percentiles_{date_suffix}-refined.csv")
        elif dep == 'trips_daily':
            path = Path("data/staging/geotab_trip_daily.csv")
        elif dep == 'hase_snapshot':
            path = Path("data/processed/hase/consumos_snapshot_latest.csv.gz")
        elif dep == 'gnv_data':
            # Simular check de datos GNV
            missing.append("GNV consumption data (estacion_servicio, litros, etc.)")
            continue
        else:
            continue

        if not path.exists():
            missing.append(str(path))

    if missing:
        print(f"⚠️ {component}: Dependencias faltantes:")
        for dep in missing:
            print(f"   - {dep}")
        return False

    return True

def enrich_component(component: str, config: dict, date_suffix: str, force: bool = False) -> bool:
    """Enriquece un componente específico."""
    print(f"\n📊 Enriqueciendo {component.upper()}")
    print(f"Status: {config['status']}")

    if '⚠️ BLOQUEADO' in config['status']:
        print(f"🚫 {component} está bloqueado, saltando...")
        return False

    if '🔄 PENDIENTE' in config['status']:
        print(f"⏳ {component} aún no implementado, creando estructura...")
        # Aquí se podría implementar la lógica de cada componente
        return False

    if '🔄 ENRIQUECIDO' in config['status']:
        print(f"✅ {component} tiene enriquecimiento disponible, ejecutando...")
        # Continuar con la ejecución

    # Verificar dependencias
    if not check_dependencies(component, config['dependencies'], date_suffix):
        return False

    # Ejecutar script del componente
    script_path = config['script']
    cmd = ['python', script_path]

    if config['config']:
        cmd.extend(['--config', config['config']])

    return run_command(cmd, f"Ejecutando {component}")

def main():
    parser = argparse.ArgumentParser(description="Enriquecer todos los agentes con telemetría avanzada")
    parser.add_argument('--date', default=datetime.now().strftime('%Y-%m-%d'),
                       help='Fecha para los datos (YYYY-MM-DD)')
    parser.add_argument('--force-reprocess', action='store_true',
                       help='Forzar reprocesamiento aunque ya existan los datos')
    parser.add_argument('--component', choices=list(COMPONENTS.keys()),
                       help='Procesar solo un componente específico')

    args = parser.parse_args()

    print("🚀 PROCESO DE ENRIQUECIMIENTO DE TELEMETRÍA")
    print("=" * 50)
    print(f"📅 Fecha: {args.date}")
    print(f"🔄 Forzar reproceso: {args.force_reprocess}")

    if args.component:
        components_to_process = [args.component]
    else:
        components_to_process = list(COMPONENTS.keys())

    print(f"\n📋 Componentes a procesar: {', '.join(components_to_process)}")

    # Verificar que los datos base existan
    events_daily = Path(f"data/staging/geotab_events_daily_{args.date}-refined.csv")
    if not events_daily.exists():
        print(f"\n❌ ERROR: No se encontraron datos de eventos para {args.date}")
        print(f"Archivo esperado: {events_daily}")
        print("\nEjecuta primero:")
        print(f"python scripts/ops/ingest_geotab.py --out-suffix {args.date}-refined")
        return 1

    # Procesar cada componente
    results = {}
    for component in components_to_process:
        config = COMPONENTS[component]
        success = enrich_component(component, config, args.date, args.force_reprocess)
        results[component] = success

    # Resumen final
    print("\n" + "=" * 50)
    print("📊 RESUMEN DE ENRIQUECIMIENTO")
    print("=" * 50)

    for component, success in results.items():
        status = "✅ ÉXITO" if success else "❌ FALLÓ"
        print(f"{component:12} - {status}")

    successful = sum(results.values())
    total = len(results)

    print(f"\n🎯 Completado: {successful}/{total} componentes")

    if successful == total:
        print("\n🎉 ¡Enriquecimiento completo exitoso!")
        print("\n📈 Próximos pasos:")
        print("1. Validar métricas de negocio mejoradas")
        print("2. Reentrenar modelos con features enriquecidos")
        print("3. Implementar alertas en tiempo real")
        print("4. Generar datos sintéticos avanzados")
    else:
        print("\n⚠️ Algunos componentes fallaron. Revisa los logs arriba.")

    return 0 if successful == total else 1

if __name__ == "__main__":
    sys.exit(main())