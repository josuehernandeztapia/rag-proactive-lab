# 🧭 Arquitectura – RAG Proactive Lab

## Visión general

```mermaid
graph TD
    BotWhatsApp[Bot Postventa (WhatsApp/Twilio)] --> API[FastAPI services/api]
    Dashboard --> API
    API -->|contexto| PIA[Agente PIA]
    API -->|features| HASE[Motor HASE]
    PIA -->|escenarios TIR| TIR
    API --> Storage[(Storage / Logs)]
    API --> Guardian[(Guardian / DTC)]
    Storage --> Dashboard
    Guardian --> API
```

- **FastAPI (`services/api/api.py`)**: núcleo que expone endpoints, integra LLMs, ingestiones y orquesta a los agentes. Usa `ACTIVE_AGENTS` para habilitar Postventa o PIA de forma independiente.
- **Agente Postventa (`agents/postventa/`)**: catálogos, scripts de equivalencias y utilidades de ingestión para el bot de soporte (WhatsApp/Twilio/Make).
- **Agente PIA (`agents/pia/`)**: evalúa escenarios TIR, reglas y prompts LLM. Consume snapshots de HASE y configuraciones financieras.
- **HASE / Scoring (`agents/hase/`, `services/api/pia_utils.py`)**: provee features en tiempo real para PIA y dashboards.
- **Guardian (`agents/guardian/`)**: pipelines y reportes de monitoreo telemétrico; genera alertas y handoffs.
- **TIR**: parte del agente PIA (archivos `tir_equilibrium_*`). Calcula reestructuras y protección financiera.
- **Storage (`services/api/storage.py`)**: maneja logging, playbooks, cache de media y exportaciones. Apoya al bot postventa y al dashboard.
- **Dashboard React (`clients/dashboard/`)**: visualiza métricas de PIA/HASE/Guardian; sincroniza datos con scripts `sync-data`.
- **Bot Postventa (WhatsApp/Twilio)**: entrada principal de casos. Usa endpoints `/twilio/whatsapp`, pipelines de visión/audio y catálogos.

## Dependencias clave

| Componente | Dependencias |
| --- | --- |
| FastAPI | `agents/postventa`, `agents/pia`, `services/api/*`, catálogos (`data/`, `guardian/`), `.env`, `ACTIVE_AGENTS` |
| Postventa | Catálogos (`agents/postventa/scripts/build_parts_*`), visión (`vision_openai.py`), audio (`audio_transcribe.py`), playbooks, storage |
| PIA | `config/financial.yml`, `data/hase/`, `data/pia/`, prompts `prompts/pia/` |
| HASE | Snapshots `data/hase/`, scripts `pia_utils` |
| Guardian | `agents/guardian/scripts/*`, `config/guardian.yml`, staging Geotab |
| Storage | Archivos en `logs/`, `data/dtc_catalog.json`, `playbooks.json` |
| Dashboard | API `/pia/*`, `reports/*`, feeds `data/processed/pia/*`, `clients/dashboard/src` |

## Flujo de build/test

1. `npm run build-safe` – empaqueta el dashboard React.
2. `npm run validate` – verifica dependencias de FastAPI.
3. `npm run test-safe` – ejecuta `pytest` sobre `services/api` y utilidades.
4. `npm run build-custom` – orquesta los tres pasos anteriores automáticamente.

## Archivos críticos

- `services/api/api.py`: Pipeline principal (Twilio, `/query`, `/pia/*`), controlado por `ACTIVE_AGENTS`.
- `agents/postventa/scripts/*`: Generadores de catálogos, equivalencias y feeds para el bot.
- `agents/pia/src/*`: Reglas, prompts y motor TIR.
- `services/api/storage.py`: Persistencia, cache y playbooks.
- `data/parts_equivalences.*`, `parts_index.json`: Catálogo de refacciones postventa.
- `data/dtc_catalog.json`, `guardian/dtc_catalog.json`: Referencias DTC.
- `prompts/pia/`: Plantillas LLM para notificaciones y resúmenes.

## Notas operativas

- Nx está disponible pero el flujo recomendado usa `build-custom.js` por estabilidad.
- Ejecuta `npm run build-custom` antes de desplegar o abrir PRs.
- Mantén `sensibles.zip` actualizado con `.env` y `secrets.local.txt` para credenciales.
- Para levantar servicios rápidamente: `make run-postventa`, `make run-pia` o `make run-all` (usa `make stop` para cerrarlos).
