# Guardian · Playbook Streaming (Geotab → Make → Neon → WhatsApp)

## 1. Webhook Geotab → Make
1. Solicitar credenciales (database, user, password) y habilitar el Data Feed (LogRecord / FaultData) o webhook personalizado.
2. Configurar un scenario en Make con webhook HTTP de entrada.
3. Normalizar el payload al esquema común:
   ```json
   {
     "date_time": "2025-10-05T14:20:00-06:00",
     "device_id": "b45F8C7A3D912",
     "vehicle_name": "HINO-921",
     "license_plate": "NJK-345-C",
     "latitude": 20.6053,
     "longitude": -100.3971,
     "event_type": "Harsh Braking",
     "event_payload": { ... raw Geotab ... }
   }
   ```

## 2. Evaluación de umbrales Guardian
1. Cargar configuración dinámica (YAML/tabla) con thresholds:
   - `telemetry_health.max_minutes_without_data`
   - `geofence.safe_zones`
   - `off_hours_usage.working_window`
   - `dtc.min_repeat`
2. Para cada evento:
   - Evaluar reglas (ej. si `event_type == "Harsh Braking"` → contador en 7 días).
   - Consolidar contexto extra (último consumo, zona autorizada, driver, etc.).
3. Si se supera un umbral → construir `GuardianAlert` (misma forma que `guardian_outbox_demo.jsonl`).

## 3. Entrega y persistencia
1. Enviar el mensaje a WhatsApp/Twilio con el template Guardian.
2. Enviar copia a Slack/Email interno si se marcó severidad alta.
3. Insertar la alerta en Neon/Postgres:
   ```sql
   INSERT INTO guardian_events (
     date_time, device_id, license_plate, alert_type, severity, payload_json, status
   ) VALUES (..., 'pending');
   ```
4. (Opcional) sincronizar con archivo `reports/guardian_outbox.jsonl` para mantener compatibilidad con el dashboard actual.

## 4. Respuesta / cierre loop
1. Capturar respuesta del cliente (palabra clave `POSTVENTA`) y enrutarla al agente correspondiente.
2. Actualizar `guardian_events.status` a `acknowledged` o `resolved` cuando Postventa confirme acción.
3. Programar un job en Make/Neon para generar el `Guardian Weekly Snapshot` (conteos por tipo, placa, severidad) y dejarlo listo para el dashboard/reporte.

> Resultado: al conectar el webhook, Guardian pasa de CSV batch a near real-time sin reescribir el notifier ni el dashboard; sólo cambian las fuentes que alimentan los artefactos.
