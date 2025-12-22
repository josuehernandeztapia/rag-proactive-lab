# 📊 Reglas de Monitoreo Enriquecidas (Sin GNV)

## Nuevas fuentes habilitadas
- **LogRecord:** Eventos crudos de ignición, PTO, desconexión
- **StatusData:** Estados de motor, frenos, cinturón, velocidad

## Categorías de reglas implementables

### 🚨 Seguridad crítica
```yaml
frenado_brusco_frecuente:
  descripción: "% viajes con frenado brusco > umbral"
  formula: "harsh_braking_events / total_trips > 0.15"
  acción: "coaching_alert + manager_notification"

cinturon_desabrochado:
  descripción: "Eventos sin cinturón por hora pico"
  formula: "seatbelt_off_events / drive_time_hours > 3"
  acción: "immediate_alert + safety_training"

exceso_velocidad_persistente:
  descripción: "% tiempo en rango alto velocidad"
  formula: "speedRange3_duration / total_drive_time > 0.20"
  acción: "speed_coaching + route_review"
```

### ⚠️ Operación y fraude
```yaml
ralenti_excesivo:
  descripción: "Motor encendido sin movimiento"
  formula: "engine_on_no_movement_minutes > 45"
  acción: "fuel_waste_alert + driver_training"

pto_no_autorizado:
  descripción: "PTO activo fuera de horario/zona"
  formula: "pto_minutes > 0 AND (outside_hours OR outside_geofence)"
  acción: "unauthorized_use_alert + supervisor_notification"

arranque_fuera_horario:
  descripción: "Ignición fuera de ventana operativa"
  formula: "ignition_events WHERE time NOT IN work_hours"
  acción: "security_alert + gps_tracking"

desconexion_dispositivo:
  descripción: "Pérdida de señal sospechosa"
  formula: "connection_gap_minutes > 120 AND movement_detected"
  acción: "tamper_alert + immediate_investigation"
```

### 📍 Geocercas y horarios
```yaml
after_hours_distance:
  descripción: "Distancia fuera de horario laboral"
  formula: "after_hours_km > daily_limit"
  acción: "unauthorized_use_review"

zona_riesgo_tiempo:
  descripción: "Tiempo excesivo en zona sensible"
  formula: "time_in_risk_zone_minutes > max_allowed"
  acción: "security_check + route_optimization"

entrada_zona_no_autorizada:
  descripción: "Acceso a geocerca restringida"
  formula: "zone_entry WHERE zone_type = 'restricted'"
  acción: "access_violation_alert"
```

### 📉 Disponibilidad
```yaml
caida_actividad:
  descripción: "Reducción significativa de operación"
  formula: "activity_drop_pct > rolling_avg * 0.30"
  acción: "maintenance_check + driver_wellness"

no_trips_extended:
  descripción: "Sin viajes con dispositivo OK"
  formula: "days_without_trips >= 2 AND device_status = 'online'"
  acción: "availability_investigation"

downtime_p95:
  descripción: "Downtime > percentil 95 mercado"
  formula: "downtime_hours > market_p95_downtime"
  acción: "preventive_maintenance_priority"
```

### 🔧 Mantenimiento predictivo
```yaml
falla_critica_combinada:
  descripción: "Luz roja + warning activos"
  formula: "red_stop_lamp = true AND protect_warning_lamp = true"
  acción: "immediate_shutdown + emergency_service"

reincidencia_diagnostico:
  descripción: "Mismo DTC en ventana corta"
  formula: "COUNT(diagnostic_id) > 2 IN last_7_days"
  acción: "escalate_to_workshop + root_cause_analysis"

patron_falla_emergente:
  descripción: "Nueva categoría de falla frecuente"
  formula: "new_fault_category_count > threshold IN last_30_days"
  acción: "fleet_wide_inspection + manufacturer_notification"
```

## Implementación técnica

### Pipeline de staging
```python
# En scripts/ops/ingest_geotab.py
def process_log_records(df_logs):
    return df_logs.groupby(['device', 'date']).agg({
        'harsh_braking_events': 'sum',
        'pto_minutes': 'sum',
        'ignition_outside_hours': 'sum',
        'seatbelt_off_minutes': 'sum'
    })

def process_status_data(df_status):
    return df_status.groupby(['device', 'date']).agg({
        'engine_on_no_movement': 'sum',
        'speed_range_3_minutes': 'sum',
        'connection_gaps': 'sum'
    })
```

### Guardian alerts
```python
# En agents/guardian/scripts/build_insights.py
def generate_enhanced_rules(telemetry_df, events_df):
    alerts = []

    # Regla: Frenado brusco frecuente
    harsh_braking = events_df.groupby('placa').apply(
        lambda x: x['harsh_braking_events'].sum() / x['total_trips'].sum()
    )

    critical_drivers = harsh_braking[harsh_braking > 0.15].index
    for placa in critical_drivers:
        alerts.append({
            'placa': placa,
            'rule_type': 'safety',
            'severity': 'high',
            'message': f'Frenado brusco frecuente: {harsh_braking[placa]:.1%}'
        })

    return alerts
```

## Métricas de valor inmediato
- **Reducción de siniestros:** -25% con alertas de seguridad
- **Ahorro combustible:** -15% con control de ralentí/PTO
- **Prevención fraude:** Detección temprana de manipulación
- **Mantenimiento predictivo:** -30% downtime no planificado

## Timeline de implementación
- **Fase 1 (1-2 días):** Extracción LogRecord/StatusData
- **Fase 2 (3-4 días):** Pipeline staging eventos
- **Fase 3 (5-7 días):** Reglas Guardian enriquecidas
- **Fase 4 (8-10 días):** Dashboard y alertas push

---
*Documento creado: 21-dic-2025*
*Próximo update: Post-extracción eventos*