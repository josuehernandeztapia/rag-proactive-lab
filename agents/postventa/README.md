# Agente Postventa

Scripts y utilidades para el bot de soporte (catálogos, equivalencias, feeds).

## Scripts clave
- `agents/postventa/scripts/build_parts_catalog.py` – extrae el índice OEM desde los SSOT PDF (`PDF_FILE`).
- `agents/postventa/scripts/build_parts_equivalences.py` – consolida equivalencias (CSV) y genera artefactos en `data/processed/postventa/`.

## Datasets
- `data/processed/postventa/parts_equivalences.*`
- `data/processed/postventa/parts_catalog.json`

## Endpoints relacionados
- `/query`, `/query_hybrid`, `/twilio/whatsapp`, `/twilio/whatsapp_json` (habilítalos con `ACTIVE_AGENTS=postventa`).

## Run rápido
```bash
make run-postventa # sólo bot postventa
make stop          # detener uvicorn/ngrok
```
