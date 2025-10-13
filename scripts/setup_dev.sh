#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

info() { printf '\033[1;32m[setup]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[setup]\033[0m %s\n' "$*"; }

# 1. Python via pyenv (opcional pero recomendado)
if command -v pyenv >/dev/null 2>&1; then
  info "Usando pyenv para asegurar Python 3.11.9"
  pyenv install 3.11.9 --skip-existing || true
  pyenv local 3.11.9
else
  warn "pyenv no encontrado. Asegúrate de tener Python 3.11 disponible."
fi

# 2. Virtualenv
if [[ ! -d .venv ]]; then
  info "Creando entorno virtual .venv"
  python -m venv .venv
fi
info "Actualizando pip y requirements"
"$REPO_ROOT/.venv/bin/python" -m pip install --upgrade pip
"$REPO_ROOT/.venv/bin/python" -m pip install -r requirements.txt

# 3. Node via nvm
if command -v nvm >/dev/null 2>&1; then
  info "Instalando Node 20 con nvm"
  export NVM_DIR="$HOME/.nvm"
  # shellcheck disable=SC1090
  [ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"
  nvm install 20
  nvm use 20
else
  warn "nvm no encontrado. Usa Node 20.x manualmente."
fi

info "Instalando dependencias npm"
npm install

info "Entorno listo. Para comenzar:
  source .venv/bin/activate
  nvm use 20  # si usas nvm
  npm run build-custom"
