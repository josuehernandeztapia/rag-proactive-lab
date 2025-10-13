# 🤝 Guía de Contribución

## Estándares de commits

- Sigue [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `docs:`, etc.).
- Incluye ID de tarea si aplica (`feat(api): agrega validación [JIRA-123]`).
- Prefiere commits pequeños y descriptivos.

## Checklist antes de abrir PR

- [ ] `npm run build-custom`
- [ ] Revisión de lint si tocaste el dashboard (`cd clients/dashboard && npm run lint`)
- [ ] Actualizaste documentación relacionada (README, runbooks, prompts)
- [ ] Adjuntaste evidencia (logs, capturas) para cambios funcionales

## Estilo y convenciones

- **Python**: respeta el estilo existente (PEP 8, 120 columnas) y usa `Path/os.getenv` en vez de rutas hardcodeadas.
- **TypeScript/React**: ejecuta `npm run lint`; usa componentes y hooks existentes como referencia.
- Nomenclatura: snake_case en Python, camelCase/PascalCase en TS.

## Flujo sugerido

1. `git checkout -b feat/nueva-funcionalidad`
2. Desarrolla y corre `npm run build-custom`
3. Documenta cambios relevantes en `docs/` o README
4. Abre PR con descripción clara y checklist
5. Responde feedback en ≤ 24h

## Tests útiles

- `npm run build-custom` (suite completa)
- `npm run test-safe` (solo pytest)
- `pytest tests/test_main_helpers.py::DetectTopicSwitchTests::test_keyword_triggers_switch`
- `cd clients/dashboard && npm test` (para specs de frontend cuando existan)

## Scripts de soporte

- `npm run build-safe`: build del dashboard únicamente
- `npm run validate`: verificación rápida FastAPI
- `scripts/setup_dev.sh`: bootstrap de entorno (venv + npm install)

¡Gracias por contribuir al laboratorio!
