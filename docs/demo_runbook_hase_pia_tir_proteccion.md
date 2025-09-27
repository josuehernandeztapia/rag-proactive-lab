# Demo Runbook — HASE / PIA / TIR / Protección

Esta guía describe cómo correr el demo sintético end-to-end y cómo validar cada componente (HASE, PIA, Protección y cálculo de TIR). Al final se listan notas sobre integraciones externas opcionales (WhatsApp/Twilio/Make) para escenarios más realistas.

---

## 1. Preparar el entorno
- Repositorio en `main` con los scripts actualizados.
- Python 3.10+ y dependencias ya instaladas (`pip install -r requirements.txt`).
- No se requieren claves externas para el flujo sintético, salvo que quieras generar alertas vía LLM (usa modo `template`).

### Archivos que produce el pipeline
| Archivo | Descripción |
| --- | --- |
| `data/pia/synthetic_driver_states.csv` | Cartera sintética (~200 financiamientos) con cobertura, telemetría, pagos y banderas HASE/PIA. |
| `data/pia/pia_outcomes_log.csv` | Log detallado de decisiones PIA y escenarios de protección evaluados. |
| `data/hase/pia_outcomes_features.csv` | Feature store para dashboards (protecciones restantes, outcomes por ventana, banderas de comportamiento). |
| `reports/pia_plan_summary.csv` | Resumen por plan (conteos, protecciones restantes promedio/mediana, alertas). |
| `reports/pia_llm_outbox.jsonl` (opcional) | Narrativas proactivas generadas por el notifier LLM en modo plantilla. |

---

## 2. Ejecutar el demo end-to-end

### Opción rápida (recomendado)
```bash
make demo-proteccion
```
Esto ejecuta internamente `scripts/demo_proteccion.py`, que a su vez:
1. Genera la cartera sintética (`pia_seed_synthetic_portfolio.py`).
2. Corre el pipeline HASE → PIA → TIR (`pia_generate_dummy_outcomes.py --reset-log`).
3. Muestra un monitor CLI (`pia_plan_summary_monitor.py`).
4. Si pasas `ARGS="--llm"`, activa el notifier LLM en modo plantilla.

Ejemplos:
- Solo datos: `make demo-proteccion`
- Demo custom: `make demo-proteccion ARGS="--size 180 --seed 99"
- Con alertas LLM: `make demo-proteccion ARGS="--llm --llm-limit 3"`

### Opción manual (paso a paso)
```bash
python3 scripts/pia_seed_synthetic_portfolio.py --size 200 --seed 2025
python3 scripts/pia_generate_dummy_outcomes.py --reset-log
python3 scripts/pia_plan_summary_monitor.py
# opcional LLM
PIA_LLM_MODE=template PIA_LLM_ALERTS=1 python3 scripts/pia_llm_notifier.py \
  --limit 3 --pia-outbox reports/pia_llm_outbox.jsonl --skip-email
```

---

## 3. Validar cada componente

### 3.1 HASE (scoring)
Verifica que el stub use los snapshots generados (cobertura, pagos, telemetría):
```bash
python3 - <<'PY'
from agents.hase.src.service import score_payload
payload = {
    "placa": "LAB-022",
    "expected_payment": 16800,
    "coverage_ratio_30d": 0.45,
    "coverage_ratio_14d": 0.38,
    "downtime_hours_30d": 120,
    "arrears_amount": 1800,
    "bank_transfer": 6000,
    "gnv_credit_30d": 7000,
}
score = score_payload(payload)
print(score.to_dict())
PY
```
> Observa que `features.protections_remaining` y el flag `used_snapshot` sean verdaderos, lo que confirma que el stub leyó `synthetic_driver_states.csv`.

### 3.2 PIA (reglas, triggers y logging)
- El script de outcomes llama internamente a `decide_action` por placa. Puedes inspeccionar un caso puntual revisando `data/pia/pia_outcomes_log.csv` o ejecutando:
```bash
python3 agents/pia/scripts/evaluate_protection_scenarios.py \
  --placa LAB-022 --market edomex --balance 470000 --payment 16800 --term 44
```
- El monitor CLI (`pia_plan_summary_monitor.py`) reporta contratos en estado crítico (expirados, revisión manual, sin protecciones).

### 3.3 Protección + TIR
- Cada outcome registrado incluye el escenario TIR seleccionado (`viable[0]`), su `annual_irr`, `payment_change`, `capitalized_interest` y si requiere revisión manual.
- Para inspeccionar a detalle usa `data/pia/pia_outcomes_log.csv` o filtra en `data/hase/pia_outcomes_features.csv` las columnas que terminan en `_synthetic_protection_*`.

### 3.4 Dashboards
Los CSV listados en la sección de fuentes se pueden conectar directamente a tu herramienta (Metabase, Superset, Power BI). En `docs/hus_dashboard_proteccion.md` tienes HUs quirúrgicas con los cortes sugeridos (riesgo vs cobertura, heatmaps de protección, composición de pagos, TIR drill-down, etc.).

---

## 4. Integraciones opcionales (WhatsApp / Twilio / Make)

El demo sintético no activa canales externos, pero el pipeline deja todo listo para conectarlos:

| Integración | Notas |
| --- | --- |
| WhatsApp Business API | Requiere Twilio/Meta + plantillas aprobadas (`PIA_OPCIONES`, `PIA_RECORDATORIO`, etc.). Se deben mapear las acciones PIA (ej. `offer_protection`) al envío de mensajes. Revisa `agents/pia/src/config.py` para nombres de plantillas. |
| Twilio | Configura `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_FROM` en `.env` y adapta `app/api.py` / `whatsapp.service.ts` para inyectar los mensajes generados por el demo. |
| Make / n8n | Cuando el notifier LLM genera alertas (`pia_llm_outbox.jsonl`), se pueden consumir desde Make; agrega un webhook que lea el JSONL o escucha eventos via `app/storage.log_event('pia_alert', …)`. |
| PWA | El `synthetic_driver_states.csv` puede hidratar stores front-end (`credit-scoring`, `risk.service`) para mostrar score HASE y banderas en el UI. También puedes mockear respuestas del backend (`/pia/protection/evaluate`) devolviendo los escenarios precomputados del log. |

> Si necesitas replicar lo que hace hoy el bot de postventa (Make + WhatsApp), usa `pia_llm_notifier.py` para generar el payload base y de allí dispara el flujo en Make. Sólo tendrás que mapear `flags`, `metric_snapshot` y `content` a tus plantillas actuales.

---

## 5. Troubleshooting rápido
- **Sin escenarios viables para muchas placas** → revisa `synthetic_driver_states.csv`; los escenarios `baseline` suelen no requerir protección. Ajusta `SCENARIO_WEIGHTS` o los parámetros de TIR si quieres mayor cobertura.
- **Score HASE siempre cero** → confirma que corriste `pia_seed_synthetic_portfolio.py` y que `synthetic_driver_states.csv` existe. El stub cachea el snapshot, reinicia el intérprete si cambias el CSV.
- **Alertas LLM vacías** → asegúrate de exportar `PIA_LLM_MODE=template` y `PIA_LLM_ALERTS=1` antes de ejecutar el notifier.

---

## 6. Checklist previo al demo en vivo
- [ ] Correr `make demo-proteccion` una hora antes y validar las rutas principales.
- [ ] Revisar el dashboard conectado a los CSV (estado > refresco). 
- [ ] (Opcional) Activar `make demo-proteccion ARGS="--llm"` para tener narrativas frescas.
- [ ] Confirmar que `data/pia/pia_outcomes_log.csv` y `data/hase/pia_outcomes_features.csv` tengan registros para las placas que usarás como ejemplo.
- [ ] Si usarás un gadget externo (WhatsApp, Make), prueba el webhook con una placa y verifica que reciba la alerta generada.

Listo: con este runbook puedes demostrar de punta a punta el funcionamiento sintético de HASE/PIA/TIR/Protección y tener un camino claro para habilitar los canales reales cuando estén disponibles.
