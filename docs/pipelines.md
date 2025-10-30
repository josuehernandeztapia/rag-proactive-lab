# Pipelines

Resumen de los flujos operativos por agente. Incluye los comandos de orquestación y los artefactos que se esperan en cada salto.

## Ingesta compartida (Geotab → staging)
1. **Copiar exportes** a `data/raw/geotab/` (Device.csv, FaultData.csv, Trip.csv, Rule.csv, Zone.csv).
2. **Normalizar** con [`scripts/ops/ingest_geotab.py`](../scripts/ops/ingest_geotab.py):
   ```bash
   python scripts/ops/ingest_geotab.py --source data/raw/geotab
   ```
3. **Salidas** en `data/staging/`:
   - `geotab_devices.csv`, `geotab_faults.csv`, `geotab_trip_daily.csv`
   - `telemetry_summary_from_geotab.csv` (downtime + activity_drop)
   - catálogos `geotab_rules.csv`, `geotab_zones.csv`

Estos staging datasets se comparten entre HASE, PIA y Guardian.

## HASE – Feature engineering y entrenamiento
1. **Features de consumo**
   ```bash
   python agents/hase/scripts/build_consumo_features.py --inputs data/raw/hase/consumos_unificados.csv
   ```
   - Escribe `data/processed/hase/consumos_features_daily.csv.gz`
   - Snapshot por placa en `data/processed/hase/consumos_snapshot_latest.csv.gz`
   - La capa agrega señales de telemetría extendida: `driving_hours_[7|14|30]d`, `seatbelt_off_rate_30d`, `high_speed_ratio_30d`, `idle_hours_ratio_30d`, `after_hours_distance_km_30d`, `telemetry_health_score`, listas para PIA/Guardian.

2. **Etiquetas dummy** (cuando no hay labels reales)
   ```bash
   python agents/hase/scripts/generate_dummy_labels.py
   ```
   - Produce `data/processed/hase/dummy_labels.csv`

3. **Dataset de entrenamiento**
   ```bash
   python agents/hase/scripts/build_training_dataset.py --labels data/processed/hase/dummy_labels.csv
   ```
   - Emite `data/processed/hase/hase_training_dataset.csv`

4. **Modelos** (opcional)
   - `python agents/hase/scripts/train_baseline.py --dataset data/processed/hase/hase_training_dataset.csv`
   - `python agents/hase/scripts/train_xgboost.py --dataset data/processed/hase/hase_training_dataset_full.csv.gz`

## PIA – Decisión y escenarios de protección
1. **Dataset base**
   ```bash
   python agents/pia/scripts/build_dataset.py
   ```
   - Carga el snapshot HASE y genera `data/processed/pia/pia_features.csv`
   - Calcula nuevos indicadores human-in-the-loop (`safety_score`, `telemetry_health_score`, `after_hours_ratio_30d`, `idle_pressure_30d`) y sugiere escenarios `safety-coaching`, `idle-rebalance`, `restructure-*`.

2. **Augment + baselines**
   ```bash
   python agents/pia/scripts/augment_dataset.py
   ```
   - Une baselines históricos y guarda `data/processed/pia/pia_features_augmented.csv`

3. **Cartera sintética**
   ```bash
   python agents/pia/scripts/pia_seed_synthetic_portfolio.py --size 200
   ```
   - Genera `data/processed/pia/synthetic_contracts.csv` y `synthetic_driver_states.csv`

4. **Outcomes y agregaciones**
   ```bash
   python agents/pia/scripts/pia_generate_dummy_outcomes.py --reset-log
   ```
   - Registra outcomes (log + features) y emite `reports/pia_plan_summary.csv`

5. **Reportes**
   ```bash
   python agents/pia/scripts/report_scenarios.py
   ```
   - Resume escenarios por plaza en `reports/pia_scenarios_summary.csv`
6. **Feed dashboard (opcional)**
   ```bash
   python agents/pia/scripts/build_pia_dashboard_hotspots.py
   ```
   - Genera `data/processed/pia/pia_hotspots.csv` con Top seguridad/telemetría para el dashboard.

## Guardian – Monitoreo de telemetría
1. Confirmar que existan staging (`geotab_faults.csv`, `telemetry_summary_from_geotab.csv`) y dataset PIA (`pia_features_augmented.csv`).
2. Ejecutar
   ```bash
   python agents/guardian/scripts/build_insights.py --config config/guardian.yml
   ```
3. Salidas:
   - `data/processed/guardian/guardian_insights.csv`
   - `reports/guardian_outbox.jsonl` (opcional, alertas para handoff)
   - Alertas nuevas: `safety` (seatbelt / alta velocidad), `idle`, `after_hours`, `telemetry`, además de `downtime`, `consumption`, `inactivity`, `driving` y `dtc`.
4. Revisión rápida: `python agents/guardian/scripts/report_hotspots.py --top 10` para identificar plazas/placas con más incidencias antes de la sesión HIL.

## Integración Postventa y TIR
- **Postventa** reutiliza `data/processed/postventa/parts_equivalences.*` para reconstruir índices (`agents/postventa/scripts/build_parts_equivalences.py`).
- **TIR** consume `synthetic_contracts.csv`/`synthetic_driver_states.csv` y los outcomes agregados para validar escenarios (ver `agents/pia/src/tir_*.py`).
- Los notebooks en `notebooks/` (e.g. `guardian_overview.ipynb`) sirven para auditorías rápidas en la Fase 4.

## Validación / Smoke
- `python scripts/smoke_test.py --base http://127.0.0.1:8000` – verifica endpoints y prompts cuando la API está arriba.
- `python agents/pia/scripts/pia_smoke_dummy_requests.py --fail-on-error` – sanity check aislado de PIA usando datasets locales.
