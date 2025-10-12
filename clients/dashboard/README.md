# Dashboard Demo Protección

Interfaz React (Vite + TypeScript) + `styled-components` para narrar los artefactos generados por `make demo-proteccion`. Consume los CSV/JSONL copiadas en `public/data` y muestra tarjetas clave, análisis de riesgo vs cobertura, drilldown de TIR y alertas LLM.

## Requisitos
- Node.js 18+
- Datasets frescos desde el repo raíz (`make demo-proteccion ARGS="--llm --llm-limit 3"`)

## Instalación
```bash
npm install
```

## Desarrollo local
```bash
npm run dev
```

El servidor de Vite quedará disponible en http://localhost:5173. Usa el botón "Refrescar datos" del dashboard tras regenerar archivos.

## Stack UI
- Theming oscuro con `styled-components` (`src/theme.ts`, `src/GlobalStyles.ts`).
- Tokens alineados al PWA (teal/amber/red/gray) y componentes base (`Card`, `StatCard`, `Badge`, `DashboardLayout`).
- Recharts + React Table para visualizaciones y drilldowns.

## Build de producción
```bash
npm run build
```

Los artefactos se publican en `dist/`.

## Actualizar datasets
1. Desde el raíz del repositorio, ejecuta:
   ```bash
   make demo-proteccion ARGS="--llm --llm-limit 3"
   ```
2. Copia (o sincroniza) los archivos generados hacia `dashboard/public/data/`. Puedes usar:
   ```bash
   npm run sync-data
   ```
   Los nombres esperados son:
   - `data/pia/synthetic_driver_states.csv`
   - `data/pia/pia_outcomes_log.csv`
   - `data/hase/pia_outcomes_features.csv`
   - `reports/pia_plan_summary.csv`
   - `reports/pia_llm_outbox.jsonl`

El dashboard usa `React Query` con fetch directo a `/data/*`. Si necesitas servirlos desde otro origen, define en `.env`:
```
VITE_DEMO_DATA_BASE=https://tu-servidor.com/data
VITE_DEMO_REPORTS_BASE=https://tu-servidor.com/data
```
De lo contrario basta recargar el navegador o presionar "Refrescar datos" para ver el último corte.

## Referencias útiles
- `../README.md` – sección **Demo Sintético Rápido**
- `../docs/demo_runbook_hase_pia_tir_proteccion.md`
- `../docs/hus_dashboard_proteccion.md`
