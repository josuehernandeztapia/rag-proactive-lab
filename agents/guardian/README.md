# Agente Guardian

Pipelines y reportes de telemetría para alertas operativas.

## Scripts clave
- `scripts/build_insights.py` – fusiona staging Geotab y dataset PIA en `data/processed/guardian/guardian_insights.csv`.
- `scripts/report_hotspots.py` – resúmenes rápidos para sesiones HIL.
- `scripts/notifier.py` – genera mensajes/handoff (opcional, reutiliza `reports/guardian_outbox.jsonl`) **y** deposita mensajes en `reports/whatsapp_outbox.jsonl`.

## Configuración
- `config/guardian.yml`: umbrales globales (puedes sobreescribir con `config/local.yaml`).
- CSV opcional para contactos: incluye columnas `placa`, `contact` (teléfono WhatsApp) y `nombre`. Úsalo con `--contacts-csv` para personalizar saludo/destinatario.

## Run rápido
```bash
make regen-guardian
python agents/guardian/scripts/report_hotspots.py --top 10
```

Para notificaciones: usa `agents/guardian/scripts/notifier.py --limit 5 --dry-run` o, sin `--dry-run`, deja mensajes en la cola unificada. Revisa `reports/whatsapp_outbox.jsonl` con `python scripts/dispatch_whatsapp_outbox.py --limit 10` antes de integrarlo con Make/Twilio.
