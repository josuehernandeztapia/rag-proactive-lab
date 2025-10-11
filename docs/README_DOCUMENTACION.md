# 📚 Índice de Documentación - RAG Proactive Lab

## 🎯 Documentación Principal

### **🚀 [README.md](../README.md)**
- Introducción completa al ecosistema
- Los 6 componentes explicados para no técnicos
- Quick start y demo sintético
- Arquitectura visual del laboratorio

## 📖 Guías por Audiencia

### **👥 Para No Técnicos**
- **[🤖 Guía Completa No Técnica](guia_no_tecnica.md)**: Qué es y cómo funciona todo el sistema
- **[📊 Componentes Completos - Sección No Técnica](componentes_completos.md)**: Explicación de cada componente

### **🔧 Para Técnicos**
- **[🏗️ Arquitectura AVI + HASE](avi_hase_scoring_architecture.md)**: Deep dive técnico de la integración
- **[📊 Componentes Completos - Sección Técnica](componentes_completos.md)**: APIs, código y arquitectura

### **🧹 Para DevOps/Mantenimiento**
- **[🗑️ Guía de Limpieza del Repositorio](repositorio_limpieza.md)**: Qué mantener y qué eliminar

## 🧠 Documentación por Componente

### **🎤 AVI - Análisis de Voz Inteligente**
- **Ubicación**: `avi_lab/README.md`
- **Código**: `avi_lab/src/app/`
- **Dataset**: `avi_lab/src/app/data/avi-questions-dataset.ts`
- **Servicios**: `avi_lab/src/app/services/voice-analysis.service.ts`

### **📊 HASE - Hyperadaptive Scoring Engine**
- **Ubicación**: `agents/hase/`
- **Modelos**: `models/hase/*.joblib`
- **Servicio**: `agents/hase/src/service.py`
- **Tests**: `agents/hase/tests/`

### **🎯 PIA - Motor de Decisión**
- **Ubicación**: `agents/pia/`
- **TIR Engine**: `agents/pia/src/tir_equilibrium_engine.py`
- **Chain Logic**: `agents/pia/src/chain.py`
- **Tests**: `agents/pia/tests/`

### **💰 TIR/Protección - Motor Financiero**
- **Configuración**: `config/financial.yml`
- **Engine**: `agents/pia/src/tir_equilibrium_engine.py`
- **Schemas**: `app/schemas/protection.py`

### **📞 Agente de Postventa**
- **Storage**: `storage.py`
- **Query Engine**: `query.py`, `query_mejorado.py`
- **Cases API**: `cases_api.py`
- **Database**: `db_cases.py`

### **📈 Dashboard React**
- **Ubicación**: `dashboard/`
- **README**: `dashboard/README.md`
- **Componentes**: `dashboard/src/components/`
- **Sync Script**: `dashboard/scripts/sync-data.mjs`

## 🔧 Documentación Técnica Especializada

### **📋 Runbooks y HUs**
- **[Demo Runbook HASE/PIA/TIR](demo_runbook_hase_pia_tir_proteccion.md)**: Guía paso a paso del demo
- **[HUs Dashboard Protección](hus_dashboard_proteccion.md)**: User stories para dashboards

### **🧪 Testing y QA**
- **Test Suite**: `tests/`
- **HASE Tests**: `agents/hase/tests/test_hase_service.py`
- **PIA Tests**: `agents/pia/tests/test_pia_agent_chain.py`
- **Storage Tests**: `tests/test_storage_*.py`

### **🔧 Configuración y Setup**
- **Financial Config**: `config/financial.yml`
- **Database Migrations**: `migrations/*.sql`
- **Environment Template**: `.env.example`
- **Makefile**: Comandos de demo y operación

## 📊 Datasets y Ejemplos

### **🎭 Demo Sintético**
- **Driver States**: `data/pia/synthetic_driver_states.csv`
- **Outcomes Log**: `data/pia/pia_outcomes_log.csv`
- **HASE Features**: `data/hase/pia_outcomes_features.csv`
- **Plan Summary**: `reports/pia_plan_summary.csv`

### **🤖 Modelos ML**
- **XGBoost**: `models/hase/hase_xgboost_model.joblib`
- **Logistic**: `models/hase/hase_logistic_baseline.joblib`
- **Métricas**: `models/hase/hase_*_metrics.json`

## 🚀 Quick Reference

### **⚡ Primera Vez Aquí?**
👉 **[QUICK_START.md](../QUICK_START.md)** - Guía de 5 minutos con troubleshooting

### **🎬 Demo Rápido**
```bash
# Ejecutar demo completo
make demo-proteccion

# Ver resultados
python3 scripts/pia_plan_summary_monitor.py

# Dashboard
cd dashboard && npm run dev
```

### **🎯 Flujo de Lectura para Humanos**

#### **🚀 Empezando (5 min)**
1. **[QUICK_START.md](../QUICK_START.md)** - Setup sin fricción
2. **[README.md](../README.md)** - Visión general de los 6 agentes

#### **📖 Entendiendo (15 min)**
3. **[Guía No Técnica](guia_no_tecnica.md)** - Qué hace cada agente
4. **[Componentes Completos](componentes_completos.md)** - Deep dive

#### **🔧 Implementando (30+ min)**
5. **[Demo Runbook](demo_runbook_hase_pia_tir_proteccion.md)** - Paso a paso
6. **[Arquitectura AVI + HASE](avi_hase_scoring_architecture.md)** - Técnico

### **🧪 Testing**
```bash
# Tests completos
pytest tests/

# Tests específicos
pytest tests/test_protection_context.py
pytest agents/hase/tests/test_hase_service.py
```

### **📊 Data Sync**
```bash
# Sincronizar dashboard
cd dashboard && npm run sync-data

# Generar datos sintéticos
python3 scripts/pia_generate_dummy_outcomes.py
```

## 🎯 Flujo de Lectura Recomendado

### **Para Entender el Sistema (No Técnico)**
1. [README.md](../README.md) - Visión general
2. [Guía No Técnica](guia_no_tecnica.md) - Explicación detallada
3. [Componentes Completos](componentes_completos.md) - Cada componente

### **Para Implementar (Técnico)**
1. [Arquitectura AVI + HASE](avi_hase_scoring_architecture.md) - Core técnico
2. [Componentes Completos](componentes_completos.md) - APIs y código
3. Código específico en cada directorio

### **Para Mantener (DevOps)**
1. [Limpieza del Repositorio](repositorio_limpieza.md) - Estructura
2. Runbooks específicos
3. Tests y validación

---

## 🎪 El Ecosistema Completo

```
📚 DOCUMENTACIÓN COMPLETA
├── 👥 No Técnicos
│   ├── ¿Qué hace cada componente?
│   ├── ¿Cómo funciona el flujo?
│   └── ¿Qué beneficios aporta?
│
├── 🔧 Técnicos
│   ├── Arquitectura y APIs
│   ├── Código e implementación
│   └── Integración entre componentes
│
├── 🧪 Testing y QA
│   ├── Suite de pruebas
│   ├── Validación de modelos
│   └── Demo sintético
│
└── 📊 Datasets y Modelos
    ├── Datos sintéticos
    ├── Modelos entrenados
    └── Configuraciones
```

**Esta documentación cubre el 100% del ecosistema RAG Proactive Lab** - desde conceptos básicos hasta implementación técnica avanzada.

*¿Preguntas específicas? Cada componente tiene su propia documentación detallada.*