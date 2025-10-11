# 🔑 Setup de Credenciales - RAG Proactive Lab

## ⚠️ Situación: Credenciales Locales vs GitHub

**✅ Local (Owner)**: Tienes todo configurado
- `.env` ← Funcional con todas las APIs
- `secrets.local.txt` ← OpenAI, Pinecone, Twilio keys
- `sensibles.zip` ← Backup completo

**❌ GitHub (Otros)**: Repo público sin credenciales
- ❌ Sin `.env` (gitignored por seguridad)
- ❌ Sin `secrets.local.txt` (gitignored por seguridad)
- ❌ Sin `sensibles.zip` (gitignored por seguridad)

**¿Por qué?** Seguridad - nunca subir API keys a Git público.

---

## 🎯 3 Opciones de Setup

### Opción 1: Demo Básico (Sin APIs) 🟡

```bash
# Copiar template
cp .env.example .env
```

**✅ Funciona:**
- Demo sintético completo
- Dashboard React
- Tests y validaciones
- PIA/HASE/TIR cálculos

**❌ NO funciona:**
- Narrativas LLM
- Búsqueda vectorial Pinecone
- Integraciones WhatsApp/Twilio

---

### Opción 2: Funcionalidad Completa (Con APIs) 🟢

```bash
# Crear .env con credenciales reales
cat > .env << 'EOF'
# APIs Esenciales
OPENAI_API_KEY="sk-..."
PINECONE_API_KEY="..."
PINECONE_ENV="us-east-1-aws"
PINECONE_INDEX="ssot-higer"

# Opcionales
TWILIO_SID="..."
TWILIO_AUTH_TOKEN="..."
NGROK_DOMAIN="tu-dominio.ngrok.app"

# El resto desde template
EOF

# Agregar el resto del template
tail -n +5 .env.example >> .env
```

**✅ Funciona TODO:**
- Demo completo + LLM narrativas
- Búsqueda vectorial inteligente
- Integración WhatsApp real
- Alertas proactivas

---

### Opción 3: Restaurar Backup 🔵

```bash
# Si tienes backup local (solo owner/team core)
unzip sensibles.zip

# O usar secrets existente
cp secrets.local.txt .env
# Nota: secrets.local.txt tiene formato comentado, necesita procesamiento
```

**👑 Para el Owner (tú)**: Ya tienes todo configurado localmente.
**👥 Para Team Members**: Solicitar `sensibles.zip` de forma segura (Slack, email encriptado).

---

## 🔍 ¿Qué Credenciales Necesito?

### OpenAI (Para LLM)
1. Ir a https://platform.openai.com/
2. Crear API key
3. `OPENAI_API_KEY="sk-..."`

### Pinecone (Para Vector Search)
1. Ir a https://www.pinecone.io/
2. Crear proyecto
3. `PINECONE_API_KEY="..."`
4. `PINECONE_ENV="us-east-1-aws"`

### Twilio (Para WhatsApp - Opcional)
1. Ir a https://www.twilio.com/
2. Obtener SID y Auth Token
3. `TWILIO_SID="..."`
4. `TWILIO_AUTH_TOKEN="..."`

---

## ✅ Verificar Setup

```bash
# Test básico (debe funcionar sin APIs)
make demo-proteccion

# Test con LLM (requiere OpenAI)
make demo-proteccion ARGS="--llm --llm-limit 3"

# Test vectorial (requiere Pinecone)
python3 -c "import pinecone; print('✅ Pinecone OK')"
```

---

## 🚨 Problemas Comunes

### Error: "No API key provided"
```bash
# Verificar que .env existe
ls -la .env

# Verificar contenido
grep OPENAI_API_KEY .env
```

### Error: "Invalid API key"
```bash
# Verificar formato
echo $OPENAI_API_KEY | head -c 10
# Debe empezar con "sk-"
```

### Error: "Pinecone index not found"
```bash
# Verificar configuración Pinecone
grep PINECONE .env
```

---

## 🔄 Flujo Recomendado para Equipos

### Desarrollador Principal
1. Setup completo con todas las APIs
2. Crear `sensibles.zip` con `.env` funcional
3. Compartir de forma segura (no por Git)

### Otros Desarrolladores
1. Recibir `sensibles.zip`
2. `unzip sensibles.zip`
3. ¡Listo para trabajar!

### CI/CD
1. Usar variables de entorno del sistema
2. NUNCA hardcodear keys en código
3. Usar secrets managers (GitHub Secrets, etc.)

---

## 📁 Estructura de Archivos Sensibles

```
rag-proactive-lab/
├── .env                    # ← Principal (gitignored)
├── .env.example           # ← Template público
├── secrets.local.txt      # ← Backup alternativo (gitignored)
├── sensibles.zip          # ← Backup comprimido (gitignored)
└── .gitignore             # ← Protege archivos sensibles
```

---

## 🎯 Resumen

| Opción | Tiempo Setup | Funcionalidad | APIs Requeridas |
|--------|-------------|---------------|----------------|
| Demo Básico | 1 min | 70% | Ninguna |
| Completo | 5 min | 100% | OpenAI + Pinecone |
| Backup | 30 seg | 100% | Ya configuradas |

**Recomendación**: Empezar con Demo Básico para validar que todo funciona, luego agregar APIs para funcionalidad completa.