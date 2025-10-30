# Runbook Operativo

Checklist para regenerar el laboratorio de punta a punta sin romper dependencias entre agentes.

## Prerrequisitos
- Python 3.10+ con dependencias instaladas (`pip install -r requirements.txt`).
- Variables de entorno cargadas (`cp .env.example .env` + credenciales reales cuando se necesiten LLM/Pinecone).
- Exports de Geotab colocados en `data/raw/geotab/` y consumos históricos en `data/raw/hase/`.
- Directorio `data/processed/` vacío o con espacio suficiente (los scripts sobrescriben archivos existentes).

## Checklist

### 1. Ingesta Geotab
- [ ] Validar presencia de `Device.csv`, `FaultData.csv`, `Trip.csv`, `Rule.csv`, `Zone.csv`.
- [ ] Ejecutar `make ingest-geotab` (equivalente a `python scripts/ops/ingest_geotab.py --source data/raw/geotab`).
- [ ] Revisar que `data/staging/geotab_trip_daily.csv` y `data/staging/telemetry_summary_from_geotab.csv` tengan filas (>0).

### 2. Regenerar datasets
- [ ] `make regen-hase` (build_consumo_features → generate_dummy_labels → build_training_dataset).
  - Verificar `data/processed/hase/consumos_features_daily.csv.gz` y `consumos_snapshot_latest.csv.gz`.
  - Confirmar que existan columnas de telemetría enriquecida (`seatbelt_off_rate_30d`, `high_speed_ratio_30d`, `idle_hours_ratio_30d`, `after_hours_distance_km_30d`, `telemetry_health_score`).
- [ ] `make regen-pia-data` (build_dataset → augment_dataset → seed portfolio → outcomes → report_scenarios).
  - Confirmar `data/processed/pia/pia_features.csv` y `pia_features_augmented.csv`.
  - Revisar que el dataset incluya `safety_score`, `safety_alert`, `after_hours_ratio_30d`, `idle_pressure_30d` y `telemetry_health_score`.
- [ ] `python agents/pia/scripts/build_pia_dashboard_hotspots.py`
  - Produce `data/processed/pia/pia_hotspots.csv` con Top seguridad/telemetría.

### 3. Entrenar / simular
- [ ] Incluido en `make regen-hase` y `make regen-pia-data`; ejecútalos si no lo hiciste en el paso anterior.
- [ ] (Opcional) `python agents/hase/scripts/train_baseline.py --dataset data/processed/hase/hase_training_dataset.csv`
  - Modelo guardado en `models/hase/`.

### 4. Validar
- [ ] `make regen-guardian` (ejecuta `agents/guardian/scripts/build_insights.py`).
  - Revisar `data/processed/guardian/guardian_insights.csv`.
  - Validar alertas `safety`, `idle`, `after_hours`, `telemetry`, además de `downtime`, `consumption`, `inactivity`, `driving` y `dtc`.
- [ ] `python agents/guardian/scripts/report_hotspots.py --top 10`
  - Resume alertas por plaza y lista las placas con más incidencias; útil para decidir ajustes locales de umbral.
- [ ] `make smoke-offline` (`OFFLINE_MODE=1` + smoke básico) o `python scripts/smoke_test.py --base http://127.0.0.1:8000` con la API real.
- [ ] (Opcional) `python agents/pia/scripts/pia_smoke_dummy_requests.py --fail-on-error` para validar prompts y orquestador PIA.

> **Modo offline:** exporta `OFFLINE_MODE=1` antes de levantar la API para que `/health` y `/query_hybrid` entreguen respuestas dummy sin conectar a Pinecone/OpenAI. Útil para smoketests locales; recuerda desactivar la variable para volver al modo completo.

### 5. Human-in-the-loop / Auditoría
- [ ] Abrir `notebooks/guardian_overview.ipynb` y `notebooks/pia_decisions.ipynb` para revisar métricas clave.
  - En `guardian_overview` validar los paneles de seguridad (seatbelt/alta velocidad) y uso fuera de horario.
  - En `pia_decisions` filtrar escenarios `safety-coaching`, `idle-rebalance` y revisar `telemetry_health_score`.
- [ ] Ejecutar `python agents/pia/scripts/pia_plan_summary_monitor.py` (o el notebook) para identificar contratos expirados, manuales o con saldo negativo.
  - El monitor ahora muestra secciones de seguridad/telemetría (tops de seatbelt/velocidad y salud de telemetría) para facilitar la priorización HIL.
- [ ] Revisar la cola unificada de WhatsApp con `python scripts/dispatch_whatsapp_outbox.py --limit 5`. Ese archivo (`reports/whatsapp_outbox.jsonl`) alimentará Make/Twilio cuando se active el canal.
- [ ] Ejecutar `node clients/dashboard/scripts/sync-data.mjs` para refrescar los feeds del dashboard (`pia_hotspots.csv`, outcomes, plan summary, etc.).
- [ ] Documentar acciones correctivas (p. ej., etiquetar manualmente, rerun `make regen-pia-data`, coordinar handoff humano).
- [ ] Registrar hallazgos en el canal acordado (`reports/pia_llm_outbox.jsonl` o dashboards HIL).

## Troubleshooting rápido
| Problema | Acción |
| --- | --- |
| `FileNotFoundError` durante build_consumo_features | Asegúrate de pasar todos los CSV origen con `--inputs`. Usa glob `data/raw/hase/*.csv` si hay múltiples archivos. |
| Columns missing en scripts de HASE/PIA | Corre nuevamente `scripts/ops/ingest_geotab.py`; los nombres se sincronizan desde staging. |
| `ModuleNotFoundError: pandas` | Ejecuta `pip install -r requirements.txt` o usa la `.venv` creada por `scripts/setup_dev.sh`. |
| Smoke test falla /health | Inicia la API con `uvicorn main:app --reload` o `make run-all`. |

## Post-run
- Versiona los artefactos relevantes (`git add data/processed/...` solo si el equipo lo permite).
- Anexa metadatos (script + commit + fecha) en `metadata.json` cuando avances al Paso 3 del plan.
