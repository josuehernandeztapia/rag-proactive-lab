# 🧹 Guía de Limpieza del Repositorio

## ❓ ¿Por qué limpiar?

El repositorio actual tiene **carpetas de referencia innecesarias** que confunden y ocupan espacio sin aportar valor al laboratorio funcional.

## 🗑️ Carpetas a Eliminar

### **Carpetas de Backup/Referencia**
```bash
# Estas NO aportan funcionalidad al laboratorio
rm -rf pwa_angular-restored/    # Backup del PWA Angular
rm -rf conductores/             # Carpeta de referencia externa
rm -rf rag-pinecone/            # Subfolder redundante
rm -rf notebooks/               # Notebooks experimentales
```

### **Archivos Duplicados**
```bash
# Archivos "2.py" y similares
find . -name "*2.py" -delete
find . -name "*backup*" -delete
find . -name "*restored*" -delete
```

## ✅ Estructura Final Limpia

```
rag-proactive-lab/
├── avi_lab/                    # 🎤 Análisis de voz (standalone integrado)
├── agents/
│   ├── hase/                   # 📊 Hyperadaptive Scoring Engine
│   └── pia/                    # 🎯 Motor de decisión TIR/Protección
├── app/                        # 🌐 API FastAPI
├── dashboard/                  # 📈 Visualización React
├── scripts/                    # ⚙️ Herramientas y demos
├── data/                       # 📊 Datasets sintéticos
├── docs/                       # 📚 Documentación
├── tests/                      # 🧪 Suite de pruebas
├── config/                     # ⚙️ Configuraciones críticas
├── models/                     # 🤖 Modelos ML entrenados
├── migrations/                 # 🗄️ Scripts de base de datos
├── pwa_angular/                # 📱 PWA (submódulo)
└── README.md                   # 📖 Documentación principal
```

## 🎯 ¿Qué mantener?

### **Core Funcional** ✅
- **`avi_lab/`** - Análisis de voz integrado (no referencia externa)
- **`agents/`** - Motores HASE y PIA
- **`app/`** - API principal
- **`dashboard/`** - UI de visualización

### **Configuración Crítica** ✅
- **`config/financial.yml`** - Políticas TIR
- **`models/hase/*.joblib`** - Modelos entrenados
- **`migrations/*.sql`** - Scripts de BD

### **Submódulos Legítimos** ✅
- **`pwa_angular/`** - PWA principal (submódulo a otro repo)

## 🚫 ¿Qué NO necesitamos?

### **Referencias Externas** ❌
- **`conductores/`** - Carpeta que apunta a otro proyecto
- **`pwa_angular-restored/`** - Backup innecesario
- **`rag-pinecone/`** - Subcarpeta redundante

### **Archivos Experimentales** ❌
- **`notebooks/`** - Desarrollo experimental
- **`*2.py`** - Archivos duplicados
- **`*backup*`** - Backups temporales

## 🔧 Script de Limpieza

```bash
#!/bin/bash
# cleanup_repository.sh

echo "🧹 Limpiando repositorio RAG Proactive Lab..."

# Eliminar carpetas de referencia
echo "📁 Eliminando carpetas de referencia..."
rm -rf pwa_angular-restored/
rm -rf conductores/
rm -rf rag-pinecone/
rm -rf notebooks/

# Eliminar archivos duplicados
echo "🗃️ Eliminando archivos duplicados..."
find . -name "*2.py" -delete
find . -name "*2.md" -delete
find . -name "*backup*" -delete
find . -name "*restored*" -delete

# Limpiar cache y temporales
echo "🧽 Limpiando cache..."
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete
find . -name ".DS_Store" -delete

echo "✅ Repositorio limpiado. Estructura final:"
ls -la | grep -v "^\."
```

## 📊 Impacto de la Limpieza

### **Antes**
- 🗂️ **~50 carpetas** (incluyendo referencias)
- 💾 **~2GB** con backups y duplicados
- 😵 **Confusión** sobre qué es funcional

### **Después**
- 🗂️ **~15 carpetas** esenciales
- 💾 **~500MB** solo funcionalidad
- ✨ **Claridad** total sobre el laboratorio

## 🎯 Principio de Limpieza

> **Si una carpeta no contribuye directamente a la funcionalidad del laboratorio, no debe estar en el repositorio principal.**

### **Criterios para Mantener:**
1. ✅ **Funcional**: Contribuye al demo/laboratorio
2. ✅ **Crítico**: Necesario para operación
3. ✅ **Integrado**: Parte del flujo principal

### **Criterios para Eliminar:**
1. ❌ **Referencia**: Solo apunta a otros proyectos
2. ❌ **Backup**: Copia de seguridad temporal
3. ❌ **Experimental**: Desarrollo no consolidado

---

**Resultado**: Un repositorio **limpio, enfocado y comprensible** que cualquier desarrollador puede entender y usar inmediatamente.