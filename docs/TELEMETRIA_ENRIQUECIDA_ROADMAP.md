# 🚀 Roadmap: Telemetría Enriquecida - Expansión a Todo el Ecosistema

## Estado actual
✅ **Guardian** - Eventos enriquecidos implementados (984 alertas adicionales)

## Componentes pendientes de enriquecimiento

### 1. **PIA** (Predictive Intelligence Analytics)
**Ubicación:** `agents/pia/scripts/build_dataset.py`

**Enriquecimientos necesarios:**
- Integrar eventos de conducción en features de riesgo
- Añadir patrones de maniobras bruscas como predictores
- Incluir métricas de ralentí en análisis de eficiencia

**Archivos a actualizar:**
- `agents/pia/scripts/build_dataset.py`
- `config/pia.yml` (crear)

**Nuevas features:**
```yaml
enhanced_driving_features:
  harsh_brake_frequency: daily_count / total_trips
  harsh_maneuver_intensity: avg(acceleration_magnitude)
  idle_efficiency_ratio: idle_minutes / engine_hours
  speed_compliance: overspeed_events / total_distance
```

### 2. **HASE** (Health Analytics & Service Engine)
**Ubicación:** `agents/hase/scripts/build_*`

**Enriquecimientos necesarios:**
- Correlacionar eventos de ralentí con consumo de GNV
- Patrones de ignición para detección de fallas
- Voltaje de dispositivo como indicador de salud eléctrica

**Archivos a actualizar:**
- `agents/hase/scripts/build_consumo_features.py`
- `agents/hase/scripts/build_training_dataset.py`
- `config/hase.yml` (crear)

**Features de diagnóstico:**
```yaml
enhanced_diagnostics:
  electrical_health: device_voltage_patterns
  engine_efficiency: idle_vs_consumption_ratio
  operational_stress: harsh_events_correlation
```

**⚠️ Bloqueador crítico:** GNV data gap - necesario para features de consumo

### 3. **Dashboards PIA**
**Ubicación:** `agents/pia/scripts/build_pia_dashboard_hotspots.py`

**Enriquecimientos necesarios:**
- Hotspots por eventos de seguridad enriquecidos
- Mapas de calor de maniobras bruscas
- Zones de riesgo por exceso de velocidad

### 4. **Postventa** (Parts & Service)
**Ubicación:** `agents/postventa/scripts/build_*`

**Enriquecimientos necesarios:**
- Correlacionar eventos de voltaje con fallas eléctricas
- Patrones de ralentí con mantenimiento preventivo
- Eventos de ignición con diagnósticos de motor

## Pipeline de ingesta enriquecido

### Fase 1: Extracción enriquecida ✅
```bash
# Geotab API con LogRecord/StatusData habilitados
python /Users/juanjosuehernandeztapia/geotab-api/get_geotab_data.py
```

### Fase 2: Procesamiento inteligente ✅
```bash
# Pipeline con clasificación de eventos refinada
python scripts/ops/ingest_geotab.py --logrecord --status --out-suffix YYYY-MM-DD
```

### Fase 3: Distribución a agentes
```bash
# Guardian ✅
python agents/guardian/scripts/build_insights.py

# PIA (pendiente)
python agents/pia/scripts/build_dataset.py --with-events

# HASE (pendiente - bloqueado por GNV)
python agents/hase/scripts/build_consumo_features.py --with-events

# Dashboards (pendiente)
python agents/pia/scripts/build_pia_dashboard_hotspots.py --with-events
```

## Configuraciones estándar

### Template de configuración por agente:
```yaml
# config/{agent}.yml
timezone: America/Mexico_City
paths:
  devices: data/staging/geotab_devices.csv
  events_daily: data/staging/geotab_events_daily_{date}_refined.csv
  events_percentiles: data/staging/geotab_events_market_percentiles_{date}_refined.csv
  output: data/processed/{agent}/{agent}_insights.csv

enhanced_features:
  safety_events: [harsh_brake, harsh_maneuver, overspeed, seatbelt_off]
  operational_events: [idling, pto, device_disconnect]
  behavioral_events: [ignition_on, ignition_off, after_hours]

thresholds:
  market_based: true
  use_percentiles: [p95, p99]
  adaptive: true
```

## Datos sintéticos enriquecidos

### Generación de eventos sintéticos:
```python
# scripts/synthetic/generate_enhanced_events.py
def generate_enhanced_telemetry():
    return {
        'harsh_driving_patterns': correlate_with_weather_traffic(),
        'operational_inefficiencies': simulate_driver_fatigue(),
        'device_health_degradation': model_voltage_decay(),
        'seasonal_variations': adjust_for_climate_patterns()
    }
```

## Entrenamiento de modelos

### Modelos a reentrenar con eventos enriquecidos:

1. **PIA Risk Scoring**
   - Features: + eventos de conducción
   - Target: Predicción de siniestros
   - Mejora esperada: +15% accuracy

2. **HASE Predictive Maintenance**
   - Features: + patrones de voltaje + ignición
   - Target: Fallas próximas
   - Mejora esperada: +20% precision

3. **Efficiency Optimization**
   - Features: + ralentí inteligente
   - Target: Consumo óptimo
   - Mejora esperada: -10% consumo

## Cronograma de implementación

### Semana 1: PIA Enhancement
- [ ] Crear `config/pia.yml`
- [ ] Modificar `build_dataset.py` para eventos
- [ ] Reentrenar modelos de riesgo
- [ ] Validar mejora en predicciones

### Semana 2: HASE Enhancement (dependiente de GNV)
- [ ] Resolver gap de datos GNV
- [ ] Crear `config/hase.yml`
- [ ] Integrar eventos en features de consumo
- [ ] Correlacionar diagnósticos con eventos

### Semana 3: Dashboards & Visualization
- [ ] Actualizar hotspots con eventos enriquecidos
- [ ] Crear visualizaciones de seguridad
- [ ] Implementar alertas en tiempo real

### Semana 4: Synthetic Data & Training
- [ ] Generar datasets sintéticos enriquecidos
- [ ] Reentrenar todos los modelos
- [ ] Validar mejoras en métricas de negocio

## Métricas de éxito

### KPIs por componente:
- **Guardian**: +984 alertas accionables ✅
- **PIA**: +15% accuracy en predicción de riesgo
- **HASE**: +20% detección temprana de fallas
- **Operacional**: -15% siniestros, -10% downtime

### ROI esperado:
- **Prevención**: $150k/año en siniestros evitados
- **Eficiencia**: $100k/año en combustible ahorrado
- **Mantenimiento**: $75k/año en reparaciones preventivas

---
*Documento creado: 21-dic-2025*
*Actualizado con: Roadmap completo post-Guardian enrichment*