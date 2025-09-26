# PIA Protección – Smoke Sintético

Este flujo permite validar la cadena completa (contrato → escenario → logging → agregación) usando los datos dummy incluidos en el repo.

## 1. Preparar el entorno

```bash
# Opcional: limpiar el log de outcomes
rm -f data/pia/pia_outcomes_log.csv
```

## 2. Probar el endpoint/CLI con contratos dummy

Los datos dummy cubren planes activos, caducados y uno que requiere revisión manual (`DEMO-005`).

```bash
# Caso activo con protecciones disponibles
python3 agents/pia/scripts/evaluate_protection_scenarios.py \
  --placa DEMO-001 --market edomex --balance 520000 --payment 19000 --term 48

# Caso caducado (no debería generar escenarios viables)
python3 agents/pia/scripts/evaluate_protection_scenarios.py \
  --placa DEMO-004 --market edomex --balance 450000 --payment 17000 --term 40
```

## 3. Registrar outcomes sintéticos

Ejecuta el script incluido para recorrer todos los contratos dummy, registrar el primer escenario viable y escribir los agregados. La bandera `--reset-log` reinicia `data/pia/pia_outcomes_log.csv` antes de generar datos nuevos.

```bash
python3 scripts/pia_generate_dummy_outcomes.py --reset-log
```

El flujo genera:

- `data/pia/pia_outcomes_log.csv`: outcomes con metadatos de plan, status y banderas de revisión.
- `data/hase/pia_outcomes_features.csv`: features agregados con columnas `protections_flag_negative`, `protections_flag_expired`, `protections_flag_manual`.
- `reports/pia_plan_summary.csv`: resumen por plan (contratos, protecciones restantes promedio/mediana, banderas).

Si prefieres sólo registrar outcomes, ejecuta el script con `--skip-aggregate` y luego corre `make pia-aggregate LOG=... OUT=... SUMMARY=...` manualmente.

## 4. Smoke de endpoint

```bash
python3 scripts/pia_smoke_dummy_requests.py --fail-on-error
```

```bash
python3 scripts/pia_llm_notifier.py --limit 3 --email-to ejemplo@rag.mx,soporte@rag.mx --pia-outbox reports/pia_llm_outbox.jsonl
```

- `--email-to` envía la alerta por correo (usa `SMTP_HOST`, `SMTP_PORT`, `SMTP_USERNAME`, `SMTP_PASSWORD`, `SMTP_USE_TLS`); si no hay SMTP, guarda el mensaje en `reports/pia_llm_email_fallback.log` para reenvío manual.
- `--pia-outbox` escribe un JSONL con el contenido listo para que PIA u otro proceso lo entregue al operador; por defecto usa `reports/pia_llm_outbox.jsonl` y registra el evento (`kind=pia_alert`) en el storage.

Este script usa FastAPI TestClient para invocar `/pia/protection/evaluate` por cada contrato dummy y reporta cuántos escenarios viables se obtienen, marcando la ejecución como fallida si alguna llamada devuelve error cuando usas `--fail-on-error`.
```bash
python3 scripts/pia_llm_worker.py --features data/hase/pia_outcomes_features.csv --interval 60 --notifier-args "--limit 3 --email-to laboratorio@rag.mx --pia-outbox reports/pia_llm_outbox.jsonl"
```

Este watcher revisa el CSV cada minuto y dispara `pia_llm_notifier` cuando detecta cambios; deténlo con Ctrl+C. En producción puedes convertirlo en servicio o reemplazarlo por cron.

Para obtener un resumen narrativo (cuando `PIA_LLM_SUMMARIES=1`), puedes llamar al endpoint `POST /pia/protection/evaluate_with_summary`, que devuelve los escenarios originales más los campos `narrative` y `narrative_context`.

## 5. Verificar resultados

- `data/hase/pia_outcomes_features.csv` debe contener columnas `last_plan_type`, `last_plan_status`, `last_plan_requires_manual_review`, `protections_remaining`.
- `reports/pia_plan_summary.csv` resume los planes (`proteccion_total`, `proteccion_basica`, etc.) e incluye el conteo `contratos_manual`.

Este smoke se puede programar como job nocturno mientras no haya datos productivos. Sustituye el CSV dummy por la fuente real cuando esté disponible.

- Si exportas `PIA_LLM_BEHAVIOUR=1` (en modo plantilla u OpenAI) el pipeline etiqueta las transcripciones de audio con señales de comportamiento y las guarda en el caso (`behaviour_tags`).

- Para disparar un caso de revisión manual:
```bash
python3 agents/pia/scripts/evaluate_protection_scenarios.py \
  --placa DEMO-005 --market edomex --balance 510000 --payment 18500 --term 48 --manual-review
```

## 6. Monitoreo rápido

```bash
python3 scripts/pia_plan_summary_monitor.py
```

Imprime el resumen de planes y detalla contratos con protecciones negativas, planes expirados o marcados para revisión manual.
