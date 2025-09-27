# Dashboard Protección · HUs Quirúrgicas

## Contexto
- El demo sintético ahora se genera con `make demo-proteccion`, que encadena:
  1. `scripts/pia_seed_synthetic_portfolio.py` — 200 financiamientos con escenarios baseline, consumption_gap, fault_alert y delinquency.
  2. `scripts/pia_generate_dummy_outcomes.py --reset-log` — corre HASE → PIA → TIR y actualiza `data/pia/pia_outcomes_log.csv`, `data/hase/pia_outcomes_features.csv`, `reports/pia_plan_summary.csv`.
  3. `scripts/pia_plan_summary_monitor.py` — resumen rápido en CLI.
  4. Opcional: `scripts/pia_llm_notifier.py` en modo template (`--llm`) para generar narrativas en `reports/pia_llm_outbox.jsonl`.
- El stub de HASE (`agents/hase/src/service.py`) consume automáticamente `data/pia/synthetic_driver_states.csv`, por lo que los scores reflejan cobertura, recaudo, fallas y protecciones reales.

## Fuentes de datos disponibles
| Archivo | Uso principal |
| --- | --- |
| `data/pia/synthetic_driver_states.csv` | Estado por placa (plan, balance, cobertura, pagos, banderas HASE/PIA). |
| `data/pia/pia_outcomes_log.csv` | Decisiones PIA y escenarios TIR evaluados (log detallado). |
| `data/hase/pia_outcomes_features.csv` | Feature store para dashboards (protecciones, outcomes por ventana, banderas). |
| `reports/pia_plan_summary.csv` | Resumen por plan (conteos, protecciones restantes, alertas). |
| `reports/pia_llm_outbox.jsonl` (opcional) | Narrativas LLM por placa con flags y métricas clave. |

## Historias de Usuario (HUs)

| ID | Rol | Necesito | Para | Criterios de aceptación |
| --- | --- | --- | --- | --- |
| HU-01 | Risk Manager | ver un snapshot del portafolio por plan (Total, Básica, Light, Caduca, Ilimitada) | priorizar seguimiento y dimensionar riesgo | Dashboard muestra tarjetas con: contratos activos, protecciones promedio, TIR promedio viable, % planes expirados. Datos vienen de `pia_plan_summary.csv`; se actualiza tras `make demo-proteccion`. |
| HU-02 | Analista PIA | identificar placas con riesgo alto por cobertura o fallas | programar intervenciones proactivas | Scatter `risk_score` (Y) vs `coverage_ratio_14d` (X) usando `pia_outcomes_features.csv`. Colores por `scenario` (`synthetic_state.scenario`) y tamaño por `arrears_amount`. Tooltips incluyen `downtime_hours_30d`, `protections_remaining`. |
| HU-03 | Operaciones Protección | monitorear salud de la protección (restantes, caducadas, manuales) | coordinar reestructuras y ventilación | Heatmap cruzando `last_plan_type` vs `protections_flag_manual`, `protections_flag_expired`, `protections_flag_negative`. Incluye listado clicable con `placa`, `protections_remaining`, `days_since_last_outcome`. |
| HU-04 | CFO | evaluar impacto de escenarios TIR | validar que flexibilidad mantiene rentabilidad | Tabla `pia_outcomes_log.csv` filtrada a outcome `synthetic_protection`. Columnas: `placa`, `action` PIA, `scenario.type`, `annual_irr`, `payment_change`, `requires_manual_review`. Permite ordenar por TIR y exportar CSV. |
| HU-05 | Tesorería | comparar composición de pagos (transferencia, efectivo) vs esperado | anticipar gaps de liquidez | Stack bar por plan/plaza con `bank_transfer` y `cash_collection` (desde `synthetic_driver_states.csv`). Añadir línea con `observed_payment / expected_payment`. Resalta placas con `arrears_amount > 0`. |
| HU-06 | Customer Success | recibir alertas accionables con narrativa | contactar operadores con discurso correcto | Widget que consume `pia_llm_outbox.jsonl`: muestra fecha, `placa`, flags (`protecciones_negativas`, `plan_expirado`, etc.), texto generado. Botón descarga JSONL. Requiere habilitar `--llm` en el pipeline. |
| HU-07 | Gestión de Estrategia | trackear eficiencia del demo | medir adopción y consistencia | Sección “Runbook” con fecha/hora del último `make demo-proteccion`, seed usado y duración de cada paso (leer `logs/demo_proteccion.log` si se agrega logging o usar timestamps de archivos). |

## Pasos siguientes sugeridos
1. Integrar `make demo-proteccion` en CI/staging con cron (ej. cada noche) para mantener datasets frescos.
2. Configurar dashboard (Superset/Metabase/Power BI) apuntando directo a los CSV/Parquet anteriores.
3. Habilitar exportaciones Parquet diarias (ej. `exports/demo/YYYYMMDD/*.parquet`) si se necesita histórico.
4. Conectar el widget LLM únicamente cuando existan alertas nuevas (comparar timestamp más reciente). 
