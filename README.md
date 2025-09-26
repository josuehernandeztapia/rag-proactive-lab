# RAG Proactive Lab

Laboratorio integrado que reúne:

- **PIA** – agente conversacional/postventa con memoria de casos, prompts híbridos y gestión de evidencias.
- **Motor TIR / Protección** – cálculo determinístico de escenarios (defer, stepdown, balloon) que preservan la TIR mínima.
- **HASE** – motor de scoring que consume outcomes + señales de comportamiento.
- **Capa LLM** – notas de expediente, alertas proactivas, storytelling y extracción de señales en texto/audio.

## Estructura

```
app/                     # FastAPI (webhooks, endpoints, prompts híbridos)
agents/
  pia/                   # Motor TIR, reglas, LLM service, contratos
  hase/                  # Scripts ingestión/agregación/entrenamiento score
scripts/                 # Herramientas de laboratorio (ingest, notifier, worker, smoke)
prompts/llm/             # Plantillas narrativas y de comportamiento
data/                    # Datasets dummy (PIA/HASE)
reports/                 # Notas, alertas, outbox y logs de LLM
docs/                    # Guías (smoke, orquestación)
```

## Flujos principales

1. **Modo laboratorio**
   ```bash
   export PIA_LLM_MODE=template
   export PIA_LLM_CASE_NOTES=1
   export PIA_LLM_ALERTS=1
   export PIA_LLM_SUMMARIES=1
   export PIA_LLM_BEHAVIOUR=1

   python3 scripts/pia_generate_dummy_outcomes.py --reset-log
   python3 scripts/pia_smoke_dummy_requests.py --fail-on-error
   python3 scripts/pia_llm_notifier.py --limit 3 --email-to laboratorio@rag.mx --pia-outbox reports/pia_llm_outbox.jsonl
   ```

2. **Watcher / cron**
   ```bash
   python3 scripts/pia_llm_worker.py \
     --features data/hase/pia_outcomes_features.csv \
     --interval 60 \
     --notifier-args "--limit 3 --email-to laboratorio@rag.mx --pia-outbox reports/pia_llm_outbox.jsonl"
   ```

3. **Storytelling**
   ```bash
   curl -X POST http://localhost:8000/pia/protection/evaluate_with_summary \
     -H "Content-Type: application/json" \
     -d '{"market":"edomex","balance":520000,"payment":19000,"term_months":48,"metadata":{"placa":"DEMO-001"}}'
   ```

4. **Señales de comportamiento**
   - `_process_media_items` agrega `behaviour_tags/notes` cuando `PIA_LLM_BEHAVIOUR=1`.
   - `aggregate_outcomes` genera columnas `behaviour_tag_*_count`, `last_behaviour_tags`, `last_behaviour_notes` listas para HASE.

## Canales de entrega

- `scripts/pia_llm_notifier.py` envía alertas vía:
  - `--email-to`: correo (SMTP; fallback en `reports/pia_llm_email_fallback.log`).
  - `--pia-outbox`: JSONL (`reports/pia_llm_outbox.jsonl`) listo para que PIA/CRM entregue el mensaje.
- `scripts/pia_llm_worker.py` monitorea el CSV y dispara el notifier en loop.

## Componentes clave

- `app/api.py`: webhook WhatsApp, prompts híbridos, endpoints `/pia/protection/evaluate[_with_summary]`.
- `agents/pia/src/*`: motor TIR, contratos, LLM service (notas, alertas, comportamiento).
- `agents/hase/scripts/*`: ingestión/agregación entrena features + score.
- `docs/pia_protection_smoke.md`: guía paso a paso del flow sintético.

## Datos generados

- `reports/pia_case_notes/*.md` – nota de cada outcome.
- `reports/pia_llm_alerts.jsonl` – historial de alertas.
- `reports/pia_llm_outbox.jsonl` – mensajes listos para el operador.
- `data/hase/pia_outcomes_features.csv` – features agregadas con flags y comportamiento.
- `reports/pia_plan_summary.csv` – resumen por plan de protección.

## Modo OpenAI

Si quieres narrativas reales:
```bash
export OPENAI_API_KEY="sk-..."
export PIA_LLM_MODE=openai
```
(el resto del pipeline es igual; si no hay red, cae automáticamente al modo plantilla).

---

Con esto tienes una visión completa del laboratorio. A partir de aquí, puedes mantener el repo "central" con todo integrado y sincronizar sólo lo necesario a cada repo "ligero" (p. ej. el bot postventa) según lo vayas desplegando.