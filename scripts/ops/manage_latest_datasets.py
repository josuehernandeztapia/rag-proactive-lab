#!/usr/bin/env python3
"""Gestión de datasets "latest" con versionado seguro.

Usage:
    python scripts/ops/manage_latest_datasets.py --promote 2025-12-21-refined
    python scripts/ops/manage_latest_datasets.py --list-versions
    python scripts/ops/manage_latest_datasets.py --rollback 2025-12-20-refined

Mantiene versioning completo + symlinks "latest" para dashboards/agentes.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict
import json
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]

# Definir datasets que necesitan "latest" symlinks
DATASETS = {
    # Staging datasets
    'events_daily': {
        'pattern': 'geotab_events_daily_*-refined.csv',
        'base_dir': 'data/staging',
        'latest_name': 'geotab_events_daily_latest.csv'
    },
    'events_percentiles': {
        'pattern': 'geotab_events_market_percentiles_*-refined.csv',
        'base_dir': 'data/staging',
        'latest_name': 'geotab_events_market_percentiles_latest.csv'
    },
    'events_raw': {
        'pattern': 'geotab_events_raw_*-refined.csv',
        'base_dir': 'data/staging',
        'latest_name': 'geotab_events_raw_latest.csv'
    },
    'trip_daily': {
        'pattern': 'geotab_trip_daily_*-refined.csv',
        'base_dir': 'data/staging',
        'latest_name': 'geotab_trip_daily_latest.csv'
    },
    'devices': {
        'pattern': 'geotab_devices_*-refined.csv',
        'base_dir': 'data/staging',
        'latest_name': 'geotab_devices_latest.csv'
    },

    # Processed datasets
    'guardian_insights': {
        'pattern': 'guardian_insights.csv',
        'base_dir': 'data/processed/guardian',
        'latest_name': 'guardian_insights_latest.csv'
    },
    'hase_insights': {
        'pattern': 'hase_insights_enhanced.csv',
        'base_dir': 'data/processed/hase',
        'latest_name': 'hase_insights_latest.csv'
    },
    'pia_features': {
        'pattern': 'pia_features_enhanced.csv',
        'base_dir': 'data/processed/pia',
        'latest_name': 'pia_features_latest.csv'
    }
}

def get_versioned_files(dataset_config: Dict) -> List[Path]:
    """Obtiene archivos versionados que coinciden con el patrón."""
    base_dir = ROOT / dataset_config['base_dir']
    pattern = dataset_config['pattern']

    if '*' in pattern:
        # Glob pattern
        files = list(base_dir.glob(pattern))
    else:
        # Archivo específico
        file_path = base_dir / pattern
        files = [file_path] if file_path.exists() else []

    # Ordenar por fecha de modificación (más reciente primero)
    return sorted(files, key=lambda x: x.stat().st_mtime, reverse=True)

def create_latest_symlink(dataset_name: str, target_file: Path) -> bool:
    """Crea symlink 'latest' apuntando al archivo especificado."""
    dataset_config = DATASETS[dataset_name]
    base_dir = ROOT / dataset_config['base_dir']
    latest_path = base_dir / dataset_config['latest_name']

    # Remover symlink existente si existe
    if latest_path.is_symlink() or latest_path.exists():
        latest_path.unlink()

    try:
        # Crear symlink relativo
        relative_target = target_file.name
        latest_path.symlink_to(relative_target)
        return True
    except Exception as e:
        print(f"❌ Error creando symlink para {dataset_name}: {e}")
        return False

def promote_version(version_suffix: str) -> bool:
    """Promociona una versión específica como 'latest'."""
    print(f"🔄 Promoviendo versión '{version_suffix}' como latest...")

    success_count = 0
    total_count = 0

    for dataset_name, config in DATASETS.items():
        total_count += 1

        # Buscar archivo con el sufijo especificado
        base_dir = ROOT / config['base_dir']
        pattern = config['pattern']

        if '*' in pattern:
            # Reemplazar * con version_suffix
            target_pattern = pattern.replace('*', version_suffix)
            target_file = base_dir / target_pattern
        else:
            # Para archivos sin versión, usar tal como está
            target_file = base_dir / pattern

        if target_file.exists():
            if create_latest_symlink(dataset_name, target_file):
                print(f"✅ {dataset_name}: {target_file.name} -> {config['latest_name']}")
                success_count += 1
            else:
                print(f"❌ {dataset_name}: Falló symlink")
        else:
            print(f"⚠️ {dataset_name}: No encontrado - {target_file}")

    print(f"\n🎯 Promovidos: {success_count}/{total_count} datasets")

    # Registrar promoción
    log_promotion(version_suffix, success_count, total_count)

    return success_count == total_count

def list_versions() -> None:
    """Lista todas las versiones disponibles."""
    print("📋 Versiones disponibles por dataset:\n")

    for dataset_name, config in DATASETS.items():
        print(f"🔹 {dataset_name}:")
        files = get_versioned_files(config)

        if files:
            for i, file_path in enumerate(files):
                # Indicar cuál es latest actualmente
                latest_path = ROOT / config['base_dir'] / config['latest_name']
                is_latest = ""

                if latest_path.is_symlink():
                    latest_target = latest_path.resolve()
                    if latest_target == file_path:
                        is_latest = " 👈 LATEST"

                # Mostrar info del archivo
                size_mb = file_path.stat().st_size / (1024 * 1024)
                mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                print(f"  {i+1}. {file_path.name} ({size_mb:.1f}MB, {mtime.strftime('%Y-%m-%d %H:%M')}){is_latest}")
        else:
            print("  (No hay archivos)")
        print()

def log_promotion(version_suffix: str, success_count: int, total_count: int) -> None:
    """Registra la promoción en el log."""
    log_dir = ROOT / "logs"
    log_dir.mkdir(exist_ok=True)

    log_file = log_dir / "dataset_promotions.jsonl"

    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "version_promoted": version_suffix,
        "success_count": success_count,
        "total_count": total_count,
        "success_rate": success_count / total_count if total_count > 0 else 0
    }

    with open(log_file, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

def rollback_version(target_version: str) -> bool:
    """Hace rollback a una versión anterior."""
    print(f"🔄 Haciendo rollback a versión '{target_version}'...")
    return promote_version(target_version)

def auto_promote_latest() -> bool:
    """Promociona automáticamente la versión más reciente encontrada."""
    print("🔄 Auto-promoviendo la versión más reciente...")

    # Buscar la versión más reciente basándose en eventos diarios
    events_daily_config = DATASETS['events_daily']
    files = get_versioned_files(events_daily_config)

    if not files:
        print("❌ No se encontraron archivos de eventos para auto-promoción")
        return False

    # Extraer sufijo de versión del archivo más reciente
    latest_file = files[0]
    filename = latest_file.name

    # Extraer versión de geotab_events_daily_XXXX-refined.csv
    if filename.startswith('geotab_events_daily_') and filename.endswith('.csv'):
        # Remover prefijo "geotab_events_daily_" y sufijo ".csv"
        version_part = filename[20:-4]
        # Si termina con "-refined", usar eso como suffix
        if version_part.endswith('-refined'):
            version_suffix = version_part
        else:
            version_suffix = version_part
        return promote_version(version_suffix)
    else:
        print(f"❌ No se puede extraer versión de {filename}")
        return False

def validate_symlinks() -> None:
    """Valida que todos los symlinks latest apunten a archivos existentes."""
    print("🔍 Validando symlinks latest...\n")

    broken_count = 0

    for dataset_name, config in DATASETS.items():
        latest_path = ROOT / config['base_dir'] / config['latest_name']

        if latest_path.exists():
            if latest_path.is_symlink():
                target = latest_path.resolve()
                if target.exists():
                    size_mb = target.stat().st_size / (1024 * 1024)
                    print(f"✅ {dataset_name}: {target.name} ({size_mb:.1f}MB)")
                else:
                    print(f"❌ {dataset_name}: Symlink roto -> {target}")
                    broken_count += 1
            else:
                print(f"⚠️ {dataset_name}: No es symlink")
        else:
            print(f"❌ {dataset_name}: Latest no existe")
            broken_count += 1

    if broken_count == 0:
        print(f"\n✅ Todos los symlinks están sanos")
    else:
        print(f"\n⚠️ {broken_count} symlinks necesitan atención")

def main() -> int:
    parser = argparse.ArgumentParser(description="Gestionar datasets latest con versionado")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--promote', help='Promover versión específica como latest')
    group.add_argument('--list-versions', action='store_true', help='Listar versiones disponibles')
    group.add_argument('--rollback', help='Hacer rollback a versión anterior')
    group.add_argument('--auto-promote', action='store_true', help='Auto-promover versión más reciente')
    group.add_argument('--auto-update', action='store_true', help='Auto-promover y validar symlinks (para pipeline)')
    group.add_argument('--validate', action='store_true', help='Validar symlinks existentes')

    args = parser.parse_args()

    print("🔗 Gestión de Datasets Latest\n")

    if args.list_versions:
        list_versions()
        return 0
    elif args.validate:
        validate_symlinks()
        return 0
    elif args.promote:
        success = promote_version(args.promote)
        return 0 if success else 1
    elif args.rollback:
        success = rollback_version(args.rollback)
        return 0 if success else 1
    elif args.auto_promote:
        success = auto_promote_latest()
        return 0 if success else 1
    elif args.auto_update:
        # Combina auto-promote y validación para usar en pipeline
        print("🔄 Auto-update: promoviendo y validando...")
        success = auto_promote_latest()
        if success:
            print("\n🔍 Validando symlinks después de promoción...")
            validate_symlinks()
            return 0
        else:
            print("❌ Auto-update falló durante promoción")
            return 1

if __name__ == "__main__":
    sys.exit(main())