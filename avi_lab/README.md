# AVI Lab (PWA)

## Objetivo
- PWA independiente para experimentar, probar y entrenar el algoritmo de voz (AVI).
- Flujo demo: grabar audio → enviar al BFF (/v1/voice/evaluate-audio) → ver voiceScore/flags/decision → exportar dataset.
- **NUEVO**: Integración completa con dataset de 55 preguntas AVI con sistema de análisis de voz inteligente.

## Estructura (resumen)
- src/app/core: contratos y ApiConfig
- src/app/services: VoiceApi + AudioRecorder + VoiceAnalysis + MicroLocalAnalysis
- src/app/pages: Runner (entrevista) + Metrics (métricas API)
- src/data: Dataset completo de 55 preguntas AVI con tipado TypeScript
- scripts: Herramientas de análisis y validación del dataset

## Comandos
- Instala dependencias: npm install
- Desarrollo: npm start
- Producción: npm run build

## Configurar API
- Edita src/environments/environment.ts (apiBaseUrl, mock)

## Mock Mode
- En la UI, activa/desactiva Mock para usar respuestas locales (sin backend).

## Nuevas Características ✨

### Dataset de 55 Preguntas AVI
- **55 preguntas** organizadas en 8 categorías temáticas
- **12 preguntas críticas** (peso ≥9) para evaluación de riesgo
- **20 preguntas de alto estrés** (nivel ≥4) para análisis psicológico
- Tipado completo en TypeScript con interfaces estructuradas

### Sistema de Análisis de Voz Inteligente
- **VoiceAnalysisService**: Análisis completo de respuestas de audio
- **MicroLocalAnalysisService**: Validación de coherencia entre preguntas
- Detección automática de indicadores de estrés
- Verificación de palabras clave de veracidad
- Scoring ponderado por criticidad de pregunta

### Interfaz de Usuario Mejorada
- **Selección de preguntas** por modo: Demo, Críticas, Alto Estrés, Categoría, Todas
- **Navegación inteligente** entre preguntas con información contextual
- **Análisis en tiempo real** con métricas detalladas
- **Exportación de sesiones** completas para análisis posterior

### Herramientas de Análisis
- Script de validación del dataset (`scripts/avi-questions-analysis.js`)
- Generación automática de reportes de integridad
- Escenarios de prueba predefinidos para validación
- Estadísticas completas de distribución y cobertura

## Modos de Operación

### Demo Mode (5 preguntas)
- Preguntas representativas de cada categoría
- Ideal para demostraciones y pruebas rápidas

### Critical Questions (12 preguntas)
- Solo preguntas con peso ≥9
- Enfoque en evaluación de riesgo crediticio

### High Stress (20 preguntas)
- Preguntas con nivel de estrés ≥4
- Análisis psicológico y detección de patrones

### Category Mode
- Selección por categoría específica
- 6-8 preguntas por categoría temática

### Full Assessment (55 preguntas)
- Evaluación completa y exhaustiva
- Análisis integral de perfil crediticio

## Backend esperado
- POST /v1/voice/evaluate-audio (multipart/form-data)
- POST /v1/voice/analyze (features pre-extraídas)
- POST /v1/voice/whisper-transcribe (transcripción + words)

## Análisis y Reportes
- Ejecutar análisis: `node scripts/avi-questions-analysis.js`
- Reportes generados en: `reports/avi-analysis-YYYY-MM-DD.json`
- Validación de integridad del dataset incluida
