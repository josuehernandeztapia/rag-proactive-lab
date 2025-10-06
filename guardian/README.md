# Guardian de Flota · Arquitectura y Flujo

## Objetivo
Convertir la telemetría de Geotab en alertas proactivas (WhatsApp, dashboard) que ayuden al cliente a actuar antes de que se generen incidentes operativos o mecánicos. Guardian complementa al agente de Postventa/PIA: Guardian **vigila y notifica**; el agente postventa **reacciona y da seguimiento**.

## Componentes (AS-IS con CSV)
- **Data sources**
  - `conductores/docs/geotab_api/FaultData.csv`: eventos de falla (diagnostic.id, estado, severidades).
  - `conductores/docs/mygeotab/Advanced Diagnostic List_*.csv`: catálogo completo de DTC (Código PXXXX + descripción).
  - `data/telemetry_summary_from_geotab*.csv`: downtime / activity drop.
  - `data/hase/consumos_unificados.csv`: consumo GNV por placa.
  - `data/pia/pia_features_augmented.csv`: cobertura, protecciones, fault flags.

- **ETL / feature engineer** (`scripts/guardian/build_insights.py`)
  - Convierte los CSV en `data/guardian_insights.csv` con columnas: `placa`, `alert_type`, `severity`, `details`, `triggered_at`.
  - Llama a `guardian/dtc_catalog.json` para mapear `diagnostic.id → P-code → descripción`.

- **Motor de reglas**
  - Umbrales configurables `config/guardian.yml` (ej. `downtime_hours_threshold`, `%_consumption_drop`, `driving_events_limit`).
  - Produce objetos `GuardianAlert` listos para enviar.

- **Notifier** (`scripts/guardian/notifier.py`)
  - Genera plantillas empatícas (Markdown/JSON) y:
    - Guarda en `reports/guardian_outbox.jsonl`.
    - Invoca webhook Make/Twilio → WhatsApp (misma infraestructura que PIA/Postventa).
  - Cada alerta incluye metadata `handoff_hint` y `handoff_keyword` para que Make enrute al bot/equipo correcto (SOPORTE, POSTVENTA, LOGISTICA, PIA, COACH).
  - Lee `alerts.autohandoff` y `alerts.autohandoff_rules` de `config/guardian.yml` para marcar `auto_escalate` y canales internos (Slack, ticket, etc.).

- **Dashboard React**
  - Reutiliza componente de alertas (sección “Guardian”) consumiendo `guardian_outbox.jsonl`.

## Ejecución (modo demo)
1. Para una demo rápida sin recargar los CSV completos, usa los artefactos de `guardian/demo/`:
   - `guardian_insights_demo.csv`
   - `guardian_outbox_demo.jsonl`
   Copia esos archivos a `data/guardian/guardian_insights.csv` y `reports/guardian_outbox.jsonl` o apunta el dashboard a la ruta demo.
2. Refresca señales reales con `python3 scripts/guardian/build_insights.py` (usa rutas/umbrales definidos en `config/guardian.yml`).
3. Genera mensajes y outbox rápido con `python3 scripts/guardian/notifier.py --limit 5` (añade `--dry-run` para solo vista previa).
   - El notifier respeta `--alert-type`, `--min-severity`, mapeo de contactos y escribe en `reports/guardian_outbox.jsonl`.
   - Usa el mismo tone empatíco del style guide y deja listo el payload para Make/Twilio.

## Integración Make / Twilio / WhatsApp (igual filosofía que PIA)
1. **Webhook** `POST /guardian/alerts` registrado en Make.
2. Make enriquece (si se desea) y despacha a:
   - Twilio WhatsApp Business (notificación al operador/propietario).
   - Notion/CRM (log). 
3. Respuestas del cliente (`reply`) llegan al agente de Postventa; Guardian sólo escucha.

### Circuito