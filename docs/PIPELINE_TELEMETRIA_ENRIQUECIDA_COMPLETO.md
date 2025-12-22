# 🔄 Pipeline Completo de Telemetría Enriquecida

## 📊 Estado de Implementación

### ✅ COMPLETADO
- **Guardian**: 1745 alertas (984 enriquecidas)
- **Pipeline de extracción**: LogRecord/StatusData refinado
- **Clasificación inteligente**: 8 tipos de eventos específicos
- **Infraestructura orquestada**: `enrich_all_agents.py`

### 🔄 PENDIENTE
- **PIA**: Features de riesgo enriquecidos
- **HASE**: Correlación con consumo (bloqueado por GNV)
- **Dashboards**: Visualizaciones de eventos

### ⚠️ BLOQUEADO
- **HASE**: Requiere datos GNV (estacion_servicio, litros, etc.)

## 🏗️ Arquitectura del Pipeline

### 1. Extracción Enriquecida
```bash
# Geotab API con eventos habilitados
cd /Users/juanjosuehernandeztapia/geotab-api
python get_geotab_data.py  # LogRecord + StatusData habilitados
```

### 2. Procesamiento Inteligente
```bash
# Clasificación refinada por IDs de diagnóstico
cd /Users/juanjosuehernandeztapia/Documents/rag-pinecone
python scripts/ops/ingest_geotab.py \
    --logrecord data/raw/geotab/LogRecord.csv \
    --status data/raw/geotab/StatusData.csv \
    --out-suffix 2025-12-21-refined
```

**Salidas generadas:**
- `geotab_events_raw_*.csv` - 4.7M eventos crudos
- `geotab_events_daily_*.csv` - Agregación diaria por placa
- `geotab_events_market_percentiles_*.csv` - Umbrales adaptativos

### 3. Distribución a Agentes
```bash
# Orquestación completa
python scripts/ops/enrich_all_agents.py

# Por componente individual
python scripts/ops/enrich_all_agents.py --component guardian
python scripts/ops/enrich_all_agents.py --component pia
```

## 🎯 Clasificación de Eventos Implementada

### Eventos de Seguridad
| Evento | Origen | Criterio | Alertas Guardian |
|--------|--------|----------|------------------|
| `harsh_brake` | DiagnosticAccelerationForwardBrakingId | < -0.3g | 5+ eventos/día |
| `harsh_maneuver` | AccelerationUpDown/SideToSide | > 0.4g | 50+ eventos/día |
| `overspeed` | DiagnosticEngineRoadSpeedId | > 80 km/h | 20+ eventos/día |
| `seatbelt_off` | Seatbelt diagnostic | = 0 | 3+ eventos/día |

### Eventos Operacionales
| Evento | Origen | Criterio | Alertas Guardian |
|--------|--------|----------|------------------|
| `idling` | DiagnosticEngineSpeedId | 600-1000 RPM | 30+ eventos/día |
| `pto` | PTO diagnostic | PTO activo | 5+ eventos/día |
| `device_disconnect` | DiagnosticGoDeviceVoltageId | < 10V | Gaps > 120min |

### Eventos de Comportamiento
| Evento | Origen | Criterio | Uso |
|--------|--------|----------|-----|
| `ignition_on/off` | DiagnosticIgnition/VehicleActiveId | > 0 / = 0 | Patrones de uso |
| `after_hours` | Timestamps + horario laboral | Fuera 06:00-22:00 | Uso no autorizado |

## 🚀 Proceso de Ingesta Estándar

### Ejecución Programada Diaria
```bash
#!/bin/bash
# scripts/daily_telemetry_enrichment.sh

DATE=$(date +%Y-%m-%d)

echo "🔄 Iniciando enriquecimiento para $DATE"

# 1. Extracción desde Geotab
cd /Users/juanjosuehernandeztapia/geotab-api
python get_geotab_data.py

# 2. Mover y procesar eventos
cd /Users/juanjosuehernandeztapia/Documents/rag-pinecone
cp /Users/juanjosuehernandeztapia/geotab-api/exports/LogRecord_*.csv data/raw/geotab/LogRecord.csv
cp /Users/juanjosuehernandeztapia/geotab-api/exports/StatusData_*.csv data/raw/geotab/StatusData.csv

# 3. Procesamiento con clasificación refinada
python scripts/ops/ingest_geotab.py \
    --logrecord data/raw/geotab/LogRecord.csv \
    --status data/raw/geotab/StatusData.csv \
    --out-suffix $DATE-refined

# 4. Enriquecimiento de todos los agentes
python scripts/ops/enrich_all_agents.py --date $DATE

echo "✅ Enriquecimiento completado para $DATE"
```

### Validación de Calidad
```bash
# Verificar volumen de eventos
wc -l data/staging/geotab_events_daily_*-refined.csv

# Verificar distribución de eventos
cut -d',' -f4-13 data/staging/geotab_events_daily_*-refined.csv | \
    awk -F',' '{for(i=1;i<=NF;i++) sum[i]+=$i} END {for(i=1;i<=NF;i++) print i, sum[i]}'

# Verificar alertas generadas
wc -l data/processed/guardian/guardian_insights.csv
grep "enhanced_" data/processed/guardian/guardian_insights.csv | wc -l
```

## 🔗 Integración con Modelos

### Features Enriquecidos para ML
```python
# Ejemplo: features para PIA Risk Scoring
enhanced_features = {
    # Seguridad
    'harsh_brake_frequency': events_daily['harsh_brake'] / total_trips,
    'harsh_maneuver_intensity': events_daily['harsh_maneuver'] / drive_hours,
    'overspeed_compliance': 1 - (events_daily['overspeed'] / total_distance),

    # Operacional
    'idle_efficiency': 1 - (events_daily['idling'] / engine_hours),
    'device_reliability': 1 - (events_daily['device_disconnect'] / total_days),

    # Comportamental
    'after_hours_ratio': events_daily['after_hours'] / total_events,
    'ignition_pattern_score': ignition_regularity_score
}
```

### Datos Sintéticos Avanzados
```python
# scripts/synthetic/generate_enhanced_events.py
def generate_correlated_events():
    # Correlaciones realistas
    correlations = {
        'harsh_brake_weather': 0.65,  # Más frenadas con lluvia
        'idle_traffic': 0.75,         # Más ralentí en tráfico
        'overspeed_highway': 0.80,    # Más velocidad en autopistas
        'after_hours_risk': 0.90      # Mayor riesgo fuera de horario
    }

    return synthetic_events_with_correlations(correlations)
```

## 📈 Métricas de Impacto

### Alertas Generadas (Post-Enriquecimiento)
- **Total**: 1745 alertas (+129% vs. baseline)
- **Enhanced Safety**: 683 alertas (nuevas)
- **Enhanced Operations**: 301 alertas (nuevas)
- **Traditional**: 761 alertas (baseline)

### Cobertura de Eventos
- **Antes**: 100% en "other" (sin clasificar)
- **Después**: 8 categorías específicas con umbrales adaptativos

### ROI Esperado
- **Prevención de siniestros**: -25% (alertas de seguridad)
- **Ahorro de combustible**: -15% (alertas de ralentí)
- **Detección de fraude**: +300% (eventos fuera de horario)
- **Mantenimiento predictivo**: -30% downtime no planificado

## 🔄 Próximos Pasos

### Inmediatos (Semana 1)
1. **Resolver GNV data gap** para desbloquear HASE
2. **Implementar PIA enhancement** con eventos de riesgo
3. **Validar mejora en predicciones** vs. baseline

### Mediano plazo (Semana 2-4)
1. **Dashboards enriquecidos** con hotspots de eventos
2. **Alertas en tiempo real** vía webhook/Slack
3. **Reentrenamiento de modelos** con features enriquecidos

### Largo plazo (Mes 2)
1. **Edge computing** para eventos en tiempo real
2. **Deep learning** sobre patrones de telemetría
3. **Gemelos digitales** de conductores/vehículos

---
*Pipeline documentado: 21-dic-2025*
*Guardian enriquecido: ✅ Operacional*
*Próximo: PIA Enhancement*