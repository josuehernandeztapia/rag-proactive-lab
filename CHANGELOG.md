# Higer RAG API — Changelog

# 0.3.0 (2025-09-18)

### Cambios principales

- Prompt híbrido refinado: plantillas por severidad/categoría, citación diferenciada (manual/catálogo/casos) y recordatorio explícito de evidencias pendientes.
- Garantía más transparente: logging cuando faltan datos, overrides manuales respetados y bloque “GARANTÍA PENDIENTE” en las respuestas.
- Pipeline de medios optimizado: caché por URL, validación de tipo/tamaño (JPEG/PNG/WEBP, audio comunes), dedupe y timestamps de adjuntos, y masking de VIN/placa antes de loggear o mostrar evidencias.
- Observabilidad ASR/OCR: eventos `asr_success/asr_timeout/asr_short_transcript` y `ocr_success/ocr_timeout` con métricas de duración/tamaño para detectar regresiones.
- Latencia/costo: resumen extractivo previo al LLM (`_summarize_to_limit`) y reutilización de embeddings entre `/query_hybrid` y webhooks (cache por hash + MessageSid).
- Catálogo priorizado: nombres normalizados/deduplicados en `build_parts_catalog.py` y catálogo OEM al frente cuando el usuario proporciona un código explícito.
- Casos/playbooks: tracking de evidencia requerida/proporcionada con timestamps, dedupe de adjuntos por hash/URL y nuevo endpoint `GET /admin/cases` con estado de cada contacto.
- Seguridad Twilio: validación activada por defecto, eventos `twilio_signature_failed` y logging de rechazos.
- Tooling: `make smoke-postdeploy` (verifica `/health`, `/version`, `/metrics`), reporte diario (`make daily-report`) y target `make test` (descubrimiento `tests/`).

### Ingesta

- CLI unificado `python3 scripts/ingest.py` con flags `--ocr`, `--bm25-only`, `--incremental`, `--recreate`, `--pages`. La ingesta incremental elimina únicamente las páginas afectadas y anota `build_ts` + `ingest_version` en los metadatos.

### Variables de entorno añadidas

- `SUMMARIZER_MODEL`, `SUMMARY_FORCE_LLM` — controlan el modelo de resumen y obligan a usar LLM en caso necesario.
- `MEDIA_CACHE_MAX`, `MEDIA_MAX_BYTES`, `TRANSCRIPT_SNIPPET_CHARS` — ajustes de procesamiento de medios.
- `SMOKE_BASE_URL` (opcional) para los comandos de smoke/postdeploy.

### Makefile (nuevos targets)

- `test`, `smoke-postdeploy`, `daily-report`.

### Notas de compatibilidad

- Los scripts `ingesta.py`, `ingesta_mejorada.py` e `ingesta_final.py` permanecen como wrappers hacia `scripts/ingest.py`.
- Las respuestas del webhook enmascaran VIN/placa en evidencias mostradas o registradas.

## 0.2.0 (2025-09-07)

### Cambios principales

- Nuevo endpoint híbrido `POST /query_hybrid` (BM25 + Pinecone) con mejor recuperación y fuentes ordenadas.
- Memoria conversacional por contacto en `/query_hybrid` y webhook de WhatsApp (historial breve agregado al prompt).
- Gestión de casos por contacto (case_id), con severidad agregada (critical/urgent/normal), categorías técnicas (brakes/oil/cooling/electrical/fuel/general), y playbooks.
- Playbooks por categoría (`playbooks.json`): pasos inmediatos de seguridad, checklist corto, evidencia mínima (ask_for) y sugerencia de ruta (taller/evaluar).
- Webhook WhatsApp `/twilio/whatsapp` ahora soporta medios (`NumMedia`, `MediaUrl0..9`) y adjunta al caso.
- Resumen inteligente de respuestas (no “cortes feos”): resumen + bullets + siguiente paso; límites por canal/severidad.
- Logging estructurado:
  - Local JSONL: `logs/events.jsonl` (eventos) y `logs/cases.jsonl` / `logs/cases_state.json` (casos).
  - Postgres/Neon (opcional): tablas `events` y `sources`.
  - Exportadores: `make export-csv` (CSV) y `make export-xlsx` (Excel con 2 hojas: events y sources).
- Soporte de dominio ngrok reservado (`NGROK_DOMAIN`) e inicio automático en macOS (LaunchAgent `com.higer.rag`).
- Corrección: `roman_to_int` movida a nivel de módulo para evitar NameError en `/query`.

### Variables de entorno añadidas

- Resumen y longitudes: `ANSWER_MAX_CHARS`, `WHATSAPP_MAX_CHARS`, `CRITICAL_MAX_CHARS`, `URGENT_MAX_CHARS`.
- Memoria/conversación: `HISTORY_LIMIT`.
- Casos/playbooks: `CASE_PREFIX`, `CASE_TTL_HOURS`, `PLAYBOOKS_FILE`.
- Híbrido: `HYBRID_TOP_K`, `HYBRID_ALPHA`, `HYBRID_DEDUPE_THRESHOLD`.
- Filtro Pinecone: `FORCE_MODEL_FILTER` (no forzado por defecto).
- Logging Neon (opcional): `STORAGE_BACKEND=dual|postgres`, `POSTGRES_URL`.
- Ngrok: `NGROK_DOMAIN`.

### Makefile (nuevos targets)

- `install-agent` / `uninstall-agent`: gestionar LaunchAgent.
- `logs-init`, `logs-tail`, `logs-test-db`: utilidades de logging.
- `export-csv`, `export-xlsx`: exportaciones.

### Notas de compatibilidad

- Las rutas existentes (`/query`, `/health`, `/version`, `/parts/search`, `/search`) se mantienen.
- `/query_hybrid` es compatible con requests sin `meta` (memoria y casos se activan cuando se envía `meta.contact`).
