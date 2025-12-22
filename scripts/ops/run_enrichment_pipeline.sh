#!/bin/bash
# Pipeline de Enriquecimiento Unificado
# Orquesta todo el proceso de enriquecimiento de agentes con telemetría

set -euo pipefail

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuración
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
ROOT_DIR="$( cd "$SCRIPT_DIR/../.." &> /dev/null && pwd )"
LOG_FILE="$ROOT_DIR/logs/enrichment_pipeline_$(date +%Y%m%d_%H%M%S).log"

# Crear directorio de logs si no existe
mkdir -p "$ROOT_DIR/logs"

# Función de logging
log() {
    echo -e "$1" | tee -a "$LOG_FILE"
}

# Función para manejar errores
handle_error() {
    log "${RED}❌ Error en paso: $1${NC}"
    log "${RED}Ver detalles en: $LOG_FILE${NC}"
    exit 1
}

# Banner
echo ""
log "${BLUE}🚀 PIPELINE DE ENRIQUECIMIENTO DE AGENTES${NC}"
log "${BLUE}==========================================${NC}"
log "📅 Inicio: $(date)"
log "📁 Directorio: $ROOT_DIR"
log "📋 Log: $LOG_FILE"
echo ""

cd "$ROOT_DIR"

# Paso 1: Procesar telemetría → eventos clasificados
log "${YELLOW}📊 PASO 1: Procesando telemetría y clasificando eventos...${NC}"
if python scripts/ops/ingest_geotab.py; then
    log "${GREEN}✅ Paso 1 completado: Eventos clasificados${NC}"
else
    handle_error "Procesamiento de telemetría"
fi

# Paso 2: HASE enriquecido → default predictions mejoradas
log "${YELLOW}🎯 PASO 2: Generando HASE enriquecido...${NC}"
if python agents/hase/scripts/build_enhanced_features.py --config config/hase.yml; then
    log "${GREEN}✅ Paso 2 completado: HASE enhanced default predictions${NC}"
else
    handle_error "HASE enriquecido"
fi

# Paso 3: PIA enriquecido → portfolio risk híbrido
log "${YELLOW}💰 PASO 3: Generando PIA enriquecido...${NC}"
if python agents/pia/scripts/build_enhanced_dataset.py --config config/pia.yml; then
    log "${GREEN}✅ Paso 3 completado: PIA enhanced portfolio risk${NC}"
else
    handle_error "PIA enriquecido"
fi

# Paso 4: Guardian insights → alertas enhanced
log "${YELLOW}🛡️  PASO 4: Generando Guardian enhanced insights...${NC}"
if python agents/guardian/scripts/build_insights.py --config config/guardian.yml; then
    log "${GREEN}✅ Paso 4 completado: Guardian enhanced alerts${NC}"
else
    handle_error "Guardian insights"
fi

# Paso 5: Actualizar dashboards con datos enriquecidos
log "${YELLOW}📈 PASO 5: Actualizando dashboards...${NC}"
if python agents/pia/scripts/build_pia_dashboard_hotspots.py; then
    log "${GREEN}✅ Paso 5 completado: Dashboard PIA actualizado${NC}"
else
    log "${YELLOW}⚠️ Dashboard PIA falló, continuando...${NC}"
fi

# Paso 6: Gestión de versiones
log "${YELLOW}🔧 PASO 6: Gestionando versiones de datasets...${NC}"
if python scripts/ops/manage_latest_datasets.py --auto-update; then
    log "${GREEN}✅ Paso 6 completado: Versiones actualizadas${NC}"
else
    log "${YELLOW}⚠️ Gestión de versiones falló, continuando...${NC}"
fi

# Paso 7: Validación de calidad (opcional)
if [[ "${1:-}" == "--validate" ]]; then
    log "${YELLOW}🔍 PASO 7: Validando calidad del enriquecimiento...${NC}"
    if python scripts/validation/validate_hybrid_weights_simple.py; then
        log "${GREEN}✅ Paso 7 completado: Validación de calidad exitosa${NC}"
    else
        log "${YELLOW}⚠️ Validación de calidad con warnings, revisando...${NC}"
    fi
fi

# Resumen final
echo ""
log "${GREEN}🎉 PIPELINE COMPLETADO EXITOSAMENTE${NC}"
log "${GREEN}==================================${NC}"
log "📅 Finalizado: $(date)"
log ""
log "📊 Archivos generados:"
log "   • HASE: data/processed/hase/enhanced_default_predictions.csv"
log "   • PIA: data/processed/pia/pia_features_enhanced.csv"
log "   • Guardian: data/processed/guardian/guardian_insights.csv"
log "   • Dashboard: data/processed/pia/pia_hotspots.csv"
log ""
log "🎯 Próximos pasos:"
log "   • Revisar hotspots en dashboard PIA"
log "   • Monitorear alertas Guardian enhanced"
log "   • Ejecutar validación A/B testing si es necesario"
log ""
log "📋 Log completo disponible en: $LOG_FILE"

# Mostrar estadísticas rápidas
echo ""
log "${BLUE}📊 ESTADÍSTICAS RÁPIDAS:${NC}"
if [[ -f "data/processed/pia/pia_features_enhanced.csv" ]]; then
    pia_records=$(wc -l < "data/processed/pia/pia_features_enhanced.csv")
    log "   • PIA Enhanced: $((pia_records - 1)) registros procesados"
fi

if [[ -f "data/processed/hase/enhanced_default_predictions.csv" ]]; then
    hase_records=$(wc -l < "data/processed/hase/enhanced_default_predictions.csv")
    log "   • HASE Enhanced: $((hase_records - 1)) registros procesados"
fi

if [[ -f "data/processed/pia/pia_hotspots.csv" ]]; then
    hotspots_records=$(wc -l < "data/processed/pia/pia_hotspots.csv")
    log "   • Dashboard Hotspots: $((hotspots_records - 1)) hotspots identificados"
fi

echo ""
log "${GREEN}✅ Pipeline de enriquecimiento completado exitosamente!${NC}"
exit 0