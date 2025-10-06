# Guardian · Checklist para conectar Geotab en vivo

1. **Credenciales**
   - Confirmar `database`, `userName`, `password` de Geotab API.
   - Solicitar habilitación de Webhooks / Data Feed para LogRecord y FaultData.

2. **Escenario Make**
   - Crear webhook de entrada con autenticación.
   - Normalizar payload → esquema Guardian (`device_id`, `license_plate`, `event_type`, `payload_json`).
   - Cargar thresholds desde `config/guardian.yml` o tabla Neon.

3. **Persistencia**
   - Crear tabla `guardian_events` en Neon/Postgres.
   - Insertar cada alerta con `status = 'pending'`, `severity`, `metadata`.
   - (Opcional) seguir escribiendo `reports/guardian_outbox.jsonl` para retrocompatibilidad.

4. **Notificaciones**
   - Conectar módulo WhatsApp/Twilio (misma plantilla PIA/Postventa).
   - Configurar avisos internos (Slack/Email) para severidades High.

5. **Dashboard / API**
   - Exponer endpoint o consulta Neon para que el dashboard lea `guardian_events`.
   - Validar filtros (placa, escenario, tipo, fecha) contra la nueva fuente.

6. **Pruebas funcionales**
   - Simular evento via API Geotab (Harsh Braking, DTC, Geofence) y verificar que Make → WhatsApp → Neon → Dashboard se actualiza.
   - Registrar evidencias (screenshots, mensajes) para el demo.

7. **Monitoreo & rollback**
   - Activar logging en Make y alertas de fallo del webhook.
   - Documentar procedimiento para desactivar el flujo (pausa scenario, desactivar webhook) si hay ruido.

> Con este checklist, al cliente sólo le queda habilitar las credenciales y conectar el webhook; Guardian ya tiene los componentes listos para operar.
