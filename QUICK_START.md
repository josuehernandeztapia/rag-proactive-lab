# 🚀 Quick Start Guide - RAG Proactive Lab

## ⚡ TL;DR - Dame 5 Minutos

```bash
# 1. Verificar todo está listo
make demo-proteccion

# 2. Ver qué generó
python3 agents/pia/scripts/pia_plan_summary_monitor.py

# 3. Dashboard visual
cd dashboard && npm run dev
```

**✅ Si funciona**: ¡Ya tienes el laboratorio completo corriendo!
**❌ Si falla**: Sigue la guía completa abajo 👇

---

## 🎯 ¿Qué Esperar?

### Después de `make demo-proteccion`:
- **📊 Cartera sintética**: ~200 financiamientos con datos realistas
- **🎯 Decisiones PIA**: Aprobaciones/rechazos con justificación
- **💰 Cálculos TIR**: Escenarios de protección financiera
- **📈 Feature Store**: Datos listos para HASE scoring
- **⏱️ Tiempo**: ~30-60 segundos

### Archivos Que Se Crean:
```
data/pia/synthetic_driver_states.csv     ← Cartera sintética
data/pia/pia_outcomes_log.csv           ← Log de decisiones
data/hase/pia_outcomes_features.csv     ← Features para ML
reports/pia_plan_summary.csv            ← Resumen ejecutivo
```

---

## 🔧 Setup Completo

### 1. **Prerequisites Check**

```bash
# Python 3.10+
python3 --version

# Dependencias
pip install -r requirements.txt

# Node.js (para dashboard)
node --version
npm --version
```

### 2. **🔑 CRÍTICO: Configuración de Credenciales**

⚠️ **El repositorio NO incluye credenciales por seguridad**. Necesitas:

#### Opción A: Usar Template (Demo Sin APIs)
```bash
# Copiar template para demo básico
cp .env.example .env
```
**✅ Permite**: Demo sintético, dashboard, tests
**❌ NO permite**: LLM narrativas, integraciones externas

#### Opción B: Credenciales Reales (Funcionalidad Completa)
```bash
# Necesitas en .env:
OPENAI_API_KEY="sk-..."      # Para LLM narrativas
PINECONE_API_KEY="..."       # Para vector search
TWILIO_SID="..."             # Para WhatsApp (opcional)
```

#### Opción C: Restaurar desde Backup
```bash
# Si tienes sensibles.zip o secrets.local.txt
unzip sensibles.zip          # Restaura .env completo
# o
# copia tu secrets.local.txt al directorio raíz
```

**🚨 Sin credenciales → Demo funciona, pero sin LLM ni búsqueda vectorial**

👉 **[SETUP_CREDENCIALES.md](SETUP_CREDENCIALES.md) - Guía detallada de configuración**

### 3. **Verificación de Estructura**

```bash
# Verifica que tienes los componentes principales
ls -la | grep -E "(avi_lab|agents|dashboard|guardian|scripts)"
```

**✅ Deberías ver**:
- `agents/postventa/` – Bot postventa (webhooks, catálogos, scripts)
- `agents/pia/` – Motor TIR + reglas de cobranza y protección
- `agents/hase/` – Ingesta/feature store de telemetría
- `agents/guardian/` – Pipelines y reportes de alertas
- `dashboard/` – React UI
- `avi_lab/`, `guardian/` – Recursos complementarios
- `scripts/` – Utilidades generales y demos

### 4. **Test Básico**

```bash
# Validar prompts
python3 scripts/validate_pia_prompts.py --verbose

# Test mínimo
python3 -c "from agents.pia.src.service import PiaAgent; print('✅ PIA OK')"
python3 -c "from agents.hase.src.service import HaseService; print('✅ HASE OK')"
```

### 5. **Levantar Solo el Agente que Necesitas**

```bash
# Postventa únicamente (webhooks WhatsApp/Make)
make run-postventa

# PIA (TIR + reglas de riesgo)
make run-pia

# API completa (endpoints postventa + PIA)
make run-all

# Detener cualquier modo
make stop
```

> También puedes fijar `ACTIVE_AGENTS` manualmente (`ACTIVE_AGENTS=postventa uvicorn main:app --reload`) si necesitas puertos personalizados o supervisores distintos.

---

## 🎬 Demo Paso a Paso

### Opción A: Demo Automático
```bash
make demo-proteccion
```

### Opción B: Paso Manual
```bash
# 1. Generar cartera sintética
python3 agents/pia/scripts/pia_seed_synthetic_portfolio.py --size 200

# 2. Procesar decisiones
python3 agents/pia/scripts/pia_generate_dummy_outcomes.py --reset-log

# 3. Ver resultados
python3 agents/pia/scripts/pia_plan_summary_monitor.py
```

### Opción C: Con Alertas LLM
```bash
make demo-proteccion ARGS="--llm --llm-limit 3"
```

---

## 📊 Dashboard React

```bash
cd dashboard

# Instalar dependencias (primera vez)
npm install

# Sincronizar datos del demo
npm run sync-data

# Levantar dashboard
npm run dev
```

**🌐 Acceder**: http://localhost:5173

**✅ Deberías ver**:
- Resumen de protecciones por plan
- Alertas de Guardian
- Métricas de HASE
- Outcomes de PIA

---

## 🔍 Troubleshooting

### ❌ Error: "ModuleNotFoundError: No module named 'yaml'"
```bash
# Instalar dependencias faltantes
pip3 install --user -r requirements.txt
# o si falla:
pip3 install --user pyyaml
```

### ❌ Error: "ModuleNotFoundError" (otros módulos)
```bash
# Agregar al PYTHONPATH
export PYTHONPATH="$PWD:$PYTHONPATH"
```

### ❌ Error: "Port 8000 already in use"
```bash
# Cambiar puerto
export PORT=8001
```

### ❌ Error: "Dashboard no carga datos"
```bash
cd dashboard
npm run sync-data
npm run dev
```

### ❌ Error: "make: command not found"
```bash
# Ejecutar manualmente
python3 scripts/demo_proteccion.py
```

### ❌ Demo toma mucho tiempo
```bash
# Demo pequeño
make demo-proteccion ARGS="--size 50"
```

---

## ✅ Validación Final

### 1. **Verificar Archivos Generados**
```bash
ls -la data/pia/synthetic_driver_states.csv
ls -la data/pia/pia_outcomes_log.csv
ls -la reports/pia_plan_summary.csv
```

### 2. **Verificar Contenido**
```bash
# Ver resumen ejecutivo
head -5 reports/pia_plan_summary.csv

# Contar decisiones
wc -l data/pia/pia_outcomes_log.csv
```

### 3. **Test Dashboard**
- ✅ Se abre en http://localhost:5173
- ✅ Muestra gráficas con datos
- ✅ No hay errores en consola

---

## 🎯 Próximos Pasos

### Para Business Users:
1. **Lee**: [Guía No Técnica](docs/guia_no_tecnica.md)
2. **Explora**: Dashboard con datos reales
3. **Entiende**: [Componentes Completos](docs/componentes_completos.md)

### Para Desarrolladores:
1. **Profundiza**: [Arquitectura AVI+HASE](docs/avi_hase_scoring_architecture.md)
2. **Tests**: `pytest tests/`
3. **Integra**: [Runbook HASE/PIA/TIR](docs/demo_runbook_hase_pia_tir_proteccion.md)

### Para DevOps:
1. **Limpieza**: `python3 scripts/ops/cleanup_repo.py`
2. **Monitoreo**: Guardian alerts
3. **Deploy**: Scripts en `scripts/ops/`

---

## 🆘 ¿Sigue Sin Funcionar?

1. **Revisa logs**: `tail -f logs/*.log`
2. **Cleanup**: `python3 scripts/ops/cleanup_repo.py`
3. **Restart fresh**:
   ```bash
   make demo-proteccion ARGS="--size 50 --seed 42"
   ```

**🔥 Hot Tip**: El 90% de problemas se resuelven con `pip install -r requirements.txt` y verificar que estás en el directorio correcto.

---

**🎯 ¡En 5 minutos deberías tener todo funcionando!**
