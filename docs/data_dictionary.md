# Data Dictionary

Guía rápida de los datasets del laboratorio y el script que los genera. Todas las rutas son relativas a la raíz del repo.

## data/raw
- `data/raw/geotab/Device.csv`, `FaultData.csv`, `Trip.csv`, `Rule.csv`, `Zone.csv` – exportes directos de Geotab. Se consumen con [`scripts/ops/ingest_geotab.py`](../scripts/ops/ingest_geotab.py).
- `data/raw/geotab/geotab_device_mapping.csv` – tabla auxiliar para mapear ids → placas durante migraciones.
- `data/raw/geotab/README.md` – notas de extracción (rangos, filtros).
- `data/raw/hase/consumos_unificados.csv` – consolidados históricos de ventas/recaudos GNV por placa. Fuente original AGS/EDOMEX.
- `data/raw/guardian/dtc_catalog.json` – catálogo extendido de códigos DTC para enriquecer alertas Guardian.

## data/staging (salida de ingestión)
- `data/staging/geotab_devices.csv` – metadata de dispositivos (placa, serie). Generado por [`scripts/ops/ingest_geotab.py`](../scripts/ops/ingest_geotab.py).
- `data/staging/geotab_faults.csv` – eventos de falla normalizados (faultState, diagnostic_id, severidad).
- `data/staging/geotab_trip_daily.csv` – agregados diarios por placa (distancia, horas, velocidades).
- `data/staging/geotab_trips_raw.csv` – respaldo de viajes crudos con los campos clave conservados.
- `data/staging/geotab_rules.csv`, `data/staging/geotab_zones.csv` – catálogos auxiliares para reglas y geocercas.
- `data/staging/telemetry_summary_from_geotab.csv` – resumen diario (downtime_hours, activity_drop_pct) usado por HASE y Guardian.

## data/processed/guardian
- `data/processed/guardian/guardian_insights.csv` – alertas accionables por placa (alert_type, severity, details, triggered_at) generadas por [`agents/guardian/scripts/build_insights.py`](../agents/guardian/scripts/build_insights.py).
- `data/processed/guardian/engine_status_summary.csv` – conteo de eventos de motor por día, utilizado para paneles.

## data/processed/hase
- `data/processed/hase/consumos_features_daily.csv.gz` – feature store diario (litros/recaudo rolling, coverage_ratio, downtime). Salida de [`agents/hase/scripts/build_consumo_features.py`](../agents/hase/scripts/build_consumo_features.py).
- `data/processed/hase/consumos_snapshot_latest.csv.gz` – último registro por placa, lista para usos en PIA/Guardian.
- Columnas clave añadidas con la nueva telemetría: `trip_count_[7|14|30]d`, `driving_hours_[7|14|30]d`, `seatbelt_off_rate_30d`, `high_speed_ratio_30d`, `idle_hours_ratio_30d`, `after_hours_distance_km_30d`, `after_hours_ratio_30d`, además de `average_speed_kph_30d` y `telemetry_health_score`.
- `data/processed/hase/consumos_summary_by_plaza.csv` – agregados estadísticos por plaza.
- `data/processed/hase/dummy_labels.csv` – etiquetas sintéticas (default_flag, reason) emitidas por [`agents/hase/scripts/generate_dummy_labels.py`](../agents/hase/scripts/generate_dummy_labels.py).
- `data/processed/hase/hase_training_dataset.csv` (+ variantes `.csv.gz`, `_full`) – dataset listo para entrenamiento HASE, construido con [`agents/hase/scripts/build_training_dataset.py`](../agents/hase/scripts/build_training_dataset.py).
- `data/processed/hase/pia_outcomes_features.csv` – features agregados desde los outcomes de PIA para consumo mixto HASE/PIA.
- `data/processed/hase/hase_components_proxy.csv`, `dummy_labels_scores.csv` – artefactos experimentales de calibración de pesos (`pre_calibrate_weights.py`).

## data/processed/pia
- `data/processed/pia/pia_features.csv` – dataset sintético base (risk_score, needs_protection, segmentos). Generado por [`agents/pia/scripts/build_dataset.py`](../agents/pia/scripts/build_dataset.py).
- `data/processed/pia/pia_features_augmented.csv` – dataset enriquecido con baselines e indicadores adicionales (`augment_dataset.py`).
- Nuevos campos relevantes expuestos por ambos datasets: `safety_score`, `safety_alert`, `telemetry_health_score`, `idle_pressure_30d`, `high_speed_pressure_30d`, `seatbelt_pressure_30d`, `after_hours_ratio_30d`, `engine_hours_30d`, `driving_hours_30d`, `idling_hours_30d`, `after_hours_distance_km_30d` y `after_hours_driving_hours_30d`.
- `data/processed/pia/hase_consumption_baselines.csv` – baseline histórico por placa (`compute_consumption_baselines.py`).
- `data/processed/pia/synthetic_contracts.csv`, `synthetic_driver_states.csv` – cartera sintética creada por [`agents/pia/scripts/pia_seed_synthetic_portfolio.py`](../agents/pia/scripts/pia_seed_synthetic_portfolio.py).
- `data/processed/pia/pia_outcomes_log.csv` – historial de outcomes generados por [`agents/pia/scripts/pia_generate_dummy_outcomes.py`](../agents/pia/scripts/pia_generate_dummy_outcomes.py).
- `data/processed/pia/protection_contracts_dummy.csv`, `protection_usage_template.csv` – plantillas para alimentar simuladores TIR/Protección.

## data/processed/postventa
- `data/processed/postventa/parts_equivalences.csv` (+ `.json`, `_validation.csv`) – mapa maestro de equivalencias de partes para el bot de postventa. Se regenera con [`agents/postventa/scripts/build_parts_equivalences.py`](../agents/postventa/scripts/build_parts_equivalences.py).

## data/processed/tir
- Reservado para artefactos financieros/equilibrio de TIR. Los scripts viven en `scripts/tir/` y emiten `data/processed/tir/*.csv` cuando se corren.

## Otros
- `data/telemetry_summary_template.csv` – layout de referencia cuando se necesita simular telemetría sin correr Geotab.
