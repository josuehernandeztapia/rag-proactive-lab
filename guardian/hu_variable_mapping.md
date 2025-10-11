# Guardian · Historias de Usuario vs Variables Geotab

| HU Guardian | Variables geotab clave | Trigger/Regla sugerida | Demo (alert_type) |
|-------------|------------------------|-------------------------|-------------------|
| Inmovilización prolongada | `idleTime`, `deviceId`, `vehicleName`, `dateTime`, `engineStatus` | `downtime_hours_30d >= 120` o `idleTime > X min` | `downtime` |
| Caída de consumo vs media | `fuelUsed`, `fuelLevel`, `distance`, `dateTime` | Diferencia vs promedio 30d/14d 25% | `consumption` |
| Salida de geocerca/resguardo | `zoneAction`, `zoneId`, `latitude`, `longitude` | `zoneAction == exit` y `tiempo fuera > max_minutes_outside` | `geofence` |
| Uso fuera de horario laboral | `tripStart`, `tripEnd`, `driverId` | Viajes entre 22:00-06:00 > umbral | `off_hours_usage` |
| Pérdida de telemetría | `deviceStatus`, `signalStrength`, `lastGPS`, `dateTime` | Sin datos > `telemetry_health.max_minutes_without_data` | `telemetry_health` |
| Código de falla con recomendación | `faultData.faultCode`, `faultDescription`, `occurrenceTime` | Map a `guardian/dtc_catalog.json`; severidad según catálogo | `dtc` |
| Hábitos de manejo riesgosos | `eventType` (harsh), `speed`, `driverId` | `harsh_event_ids` acumulados en ventana 7d | `driving` |
| Energía EV | `stateOfCharge`, `energyUsed`, `chargingStatus` | SOC < 25% o caída consumo > 30% | `energy` |
| Guardian Weekly Snapshot | Agregados de las variables anteriores | Conteos por tipo/severidad/semanas | (resumen) |
| Auditoría de alertas | `alertType`, `alertStatus`, `vehicleName`, `payload` | Guardar en Neon: `guardian_events` | (todos) |

> Cada fila usa el esquema de evento JSON como fuente y aterriza en un `alert_type` del demo. Ajusta umbrales en `config/guardian.yml` según la flota.
