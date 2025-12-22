#!/bin/bash
# Pipeline diario de enriquecimiento de telemetría
# Automatiza: Extracción -> Procesamiento -> Distribución -> Alertas

set -e  # Salir en error

DATE=$(date +%Y-%m-%d)
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${BLUE}[$(date +'%H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}✅ $1${NC}"
}

warning() {
    echo -e "${YELLOW}⚠️ $1${NC}"
}

error() {
    echo -e "${RED}❌ $1${NC}"
    exit 1
}

# Directorios base
GEOTAB_API="/Users/juanjosuehernandeztapia/geotab-api"
RAG_PINECONE="/Users/juanjosuehernandeztapia/Documents/rag-pinecone"

log "🚀 Iniciando pipeline de enriquecimiento de telemetría para $DATE"

# ==============================================================================
# FASE 1: EXTRACCIÓN DESDE GEOTAB
# ==============================================================================
log "📡 Fase 1: Extracción desde Geotab API"

cd "$GEOTAB_API" || error "No se puede acceder a $GEOTAB_API"

# Verificar que el script de extracción existe
if [ ! -f "get_geotab_data.py" ]; then
    error "Script de extracción no encontrado en $GEOTAB_API/get_geotab_data.py"
fi

log "Ejecutando extracción de Geotab..."
if python get_geotab_data.py > "logs/extraction_$TIMESTAMP.log" 2>&1; then
    success "Extracción de Geotab completada"
else
    error "Fallo en extracción de Geotab - revisa logs/extraction_$TIMESTAMP.log"
fi

# Verificar que se generaron los archivos esperados
LATEST_LOGRECORD=$(ls exports/LogRecord_*.csv 2>/dev/null | tail -1)
LATEST_STATUSDATA=$(ls exports/StatusData_*.csv 2>/dev/null | tail -1)

if [ -z "$LATEST_LOGRECORD" ] || [ -z "$LATEST_STATUSDATA" ]; then
    error "Archivos de eventos no generados correctamente"
fi

log "Archivos extraídos:"
log "  - LogRecord: $(basename "$LATEST_LOGRECORD") ($(wc -l < "$LATEST_LOGRECORD") registros)"
log "  - StatusData: $(basename "$LATEST_STATUSDATA") ($(wc -l < "$LATEST_STATUSDATA") registros)"

# ==============================================================================
# FASE 2: MOVIMIENTO Y PROCESAMIENTO
# ==============================================================================
log "🔄 Fase 2: Procesamiento inteligente de eventos"

cd "$RAG_PINECONE" || error "No se puede acceder a $RAG_PINECONE"

# Crear directorio raw si no existe
mkdir -p data/raw/geotab

# Mover archivos más recientes
log "Moviendo archivos de eventos a rag-pinecone..."
cp "$GEOTAB_API/$LATEST_LOGRECORD" data/raw/geotab/LogRecord.csv
cp "$GEOTAB_API/$LATEST_STATUSDATA" data/raw/geotab/StatusData.csv

success "Archivos movidos a data/raw/geotab/"

# Procesamiento con clasificación refinada
log "Ejecutando pipeline de clasificación de eventos..."
if python scripts/ops/ingest_geotab.py \
    --logrecord data/raw/geotab/LogRecord.csv \
    --status data/raw/geotab/StatusData.csv \
    --out-suffix "$DATE-refined" > "logs/processing_$TIMESTAMP.log" 2>&1; then
    success "Procesamiento de eventos completado"
else
    error "Fallo en procesamiento - revisa logs/processing_$TIMESTAMP.log"
fi

# Verificar salidas
EVENTS_DAILY="data/staging/geotab_events_daily_$DATE-refined.csv"
EVENTS_PERCENTILES="data/staging/geotab_events_market_percentiles_$DATE-refined.csv"

if [ ! -f "$EVENTS_DAILY" ] || [ ! -f "$EVENTS_PERCENTILES" ]; then
    error "Archivos de salida no generados correctamente"
fi

log "Eventos procesados:"
log "  - Eventos diarios: $(wc -l < "$EVENTS_DAILY") filas"
log "  - Percentiles: $(wc -l < "$EVENTS_PERCENTILES") métricas"

# ==============================================================================
# FASE 3: ENRIQUECIMIENTO DE AGENTES
# ==============================================================================
log "🤖 Fase 3: Enriquecimiento de agentes"

# Ejecutar enriquecimiento orquestado
log "Ejecutando enriquecimiento de todos los agentes..."
if python scripts/ops/enrich_all_agents.py \
    --date "$DATE" > "logs/enrichment_$TIMESTAMP.log" 2>&1; then
    success "Enriquecimiento completado"
else
    warning "Algunos agentes fallaron - revisa logs/enrichment_$TIMESTAMP.log"
fi

# ==============================================================================
# FASE 4: VALIDACIÓN Y MÉTRICAS
# ==============================================================================
log "📊 Fase 4: Validación y métricas"

# Validar Guardian insights
GUARDIAN_INSIGHTS="data/processed/guardian/guardian_insights.csv"

if [ -f "$GUARDIAN_INSIGHTS" ]; then
    TOTAL_ALERTS=$(tail -n +2 "$GUARDIAN_INSIGHTS" | wc -l)
    ENHANCED_SAFETY=$(grep -c "enhanced_safety" "$GUARDIAN_INSIGHTS" 2>/dev/null || echo 0)
    ENHANCED_OPS=$(grep -c "enhanced_operations" "$GUARDIAN_INSIGHTS" 2>/dev/null || echo 0)

    success "Guardian alertas generadas:"
    log "  - Total: $TOTAL_ALERTS alertas"
    log "  - Enhanced Safety: $ENHANCED_SAFETY alertas"
    log "  - Enhanced Operations: $ENHANCED_OPS alertas"
else
    warning "Guardian insights no generado"
fi

# Distribución de eventos por tipo
log "Distribución de eventos por tipo (muestra):"
if [ -f "$EVENTS_DAILY" ]; then
    head -1 "$EVENTS_DAILY" | tr ',' '\n' | grep -E "(harsh|idle|overspeed)" | head -5
fi

# ==============================================================================
# FASE 5: LIMPIEZA Y BACKUP
# ==============================================================================
log "🧹 Fase 5: Limpieza y backup"

# Crear backup de logs
mkdir -p backups/"$DATE"
cp logs/*"$TIMESTAMP".log backups/"$DATE"/ 2>/dev/null || true

# Limpiar archivos temporales antiguos (>7 días)
find data/raw/geotab -name "*.csv" -mtime +7 -delete 2>/dev/null || true
find logs -name "*.log" -mtime +7 -delete 2>/dev/null || true

success "Limpieza completada"

# ==============================================================================
# RESUMEN FINAL
# ==============================================================================
log "📋 Resumen del pipeline para $DATE"
echo "=========================================================================================="
echo "🕐 Duración: Inicio $(date -r $SECONDS +'%H:%M:%S') - Fin $(date +'%H:%M:%S')"
echo "📊 Eventos procesados: $([ -f "$EVENTS_DAILY" ] && echo "$(tail -n +2 "$EVENTS_DAILY" | wc -l) días/placas" || echo "N/A")"
echo "🚨 Alertas generadas: $([ -f "$GUARDIAN_INSIGHTS" ] && echo "$(tail -n +2 "$GUARDIAN_INSIGHTS" | wc -l)" || echo "N/A")"
echo "💾 Logs guardados en: backups/$DATE/"
echo "=========================================================================================="

success "🎉 Pipeline de enriquecimiento completado para $DATE"

# Notificación opcional (descomentar si se configura)
# curl -X POST "$SLACK_WEBHOOK" -d "{\"text\":\"✅ Pipeline telemetría completado: $TOTAL_ALERTS alertas generadas para $DATE\"}" || true

exit 0