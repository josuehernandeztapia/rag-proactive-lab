# 🛠️ RAG Proactive Lab – Guía de Desarrollo

Bienvenido al laboratorio. Esta guía explica cómo preparar el entorno sin depender de asistentes automáticos.

## 1. Prerrequisitos

| Herramienta | Versión recomendada | Notas |
| --- | --- | --- |
| macOS / Linux | Ventura / Ubuntu 22.04 | Otros sistemas pueden requerir ajustes |
| Git | ≥ 2.40 | `git --version` |
| Python | 3.11.9 (pyenv) | El repositorio usa `.venv` local |
| Node.js | 20.x (nvm) | Requerido para el dashboard React |
| npm | ≥ 10.0 | Se instala con Node 20 |

> **Tip**: instala [`pyenv`](https://github.com/pyenv/pyenv) y [`nvm`](https://github.com/nvm-sh/nvm) para manejar versiones.

## 2. Clonado del repositorio

```bash
mkdir -p ~/work && cd ~/work
git clone git@github.com:josuehernandeztapia/rag-proactive-lab.git
cd rag-proactive-lab
```

## 3. Configuración de Python (3.11.9)

```bash
pyenv install 3.11.9 --skip-existing
pyenv local 3.11.9
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

> Si `pyenv` no está inicializado, añade a `~/.zshrc`:
> ```bash
> export PATH="$HOME/.pyenv/bin:$PATH"
> eval "$(pyenv init -)"
> ```

## 4. Configuración de Node (20.x)

```bash
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"
nvm install 20
nvm use 20
npm install
```

## 5. Variables sensibles

- Descomprime `sensibles.zip` (contraseña compartida internamente).
- Verifica que `.env` y `secrets.local.txt` estén en la raíz.
- Nunca los subas al repositorio.

## 6. Comandos clave

| Acción | Comando |
| --- | --- |
| Build completo (React + FastAPI + tests) | `npm run build-custom` |
| Solo dashboard | `npm run build-safe` |
| Suite de pruebas | `npm run test-safe` |
| Validación rápida de FastAPI | `npm run validate` |

> `npm run build-custom` usa `.venv/bin/python`. Si tu intérprete está en otra ruta, exporta `RAG_PYTHON=/ruta/a/python` antes de ejecutar.

## 7. Flujo de trabajo recomendado

1. `source .venv/bin/activate`
2. `nvm use 20`
3. Edita código
4. `npm run build-custom`
5. Backend opcional: `uvicorn main:app --reload`
6. Dashboard: `cd clients/dashboard && npm run dev`

## 8. Troubleshooting

| Problema | Solución |
| --- | --- |
| `ModuleNotFoundError: fastapi` | Verifica `.venv` y vuelve a instalar requirements |
| `Failed to start plugin worker` | Usa los scripts custom; Nx es opcional |
| `command not found: python` | Asegura `pyenv init` en tu shell |
| Permisos sobre `build-custom.js` | `chmod +x build-custom.js` |

## 9. Convenciones de contribución

- Usa mensajes tipo Conventional Commits (`feat:`, `fix:`, `docs:`...).
- Ejecuta `npm run build-custom` antes de push.
- Añade pruebas cuando cambies lógica en `services/api/api.py` o `agents/pia/`.

## 10. Recursos

- [README principal](../README.md)
- [Visión arquitectónica](./architecture.md)
- [Guía de contribución](./CONTRIBUTING.md)

---
Para dudas, contacta al equipo de arquitectura.
