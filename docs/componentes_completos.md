# 🚀 Guía Completa de Componentes - RAG Proactive Lab

## 🎯 Los 6 Componentes del Ecosistema

Este documento explica **todos los componentes** del laboratorio de manera clara para técnicos y no técnicos.

---

## 🎤 **1. AVI - Análisis de Voz Inteligente**

### **Para No Técnicos** 👥

**¿Qué es?**
Un entrevistador virtual que analiza tu voz mientras respondes preguntas.

**¿Cómo funciona?**
- Te hace 55 preguntas sobre tu situación financiera
- Mientras hablas, analiza tu voz en tiempo real
- No graba tus palabras, solo analiza patrones

**¿Qué detecta?**
- **Estrés**: ¿Te pones nervioso con ciertas preguntas?
- **Confianza**: ¿Hablas con seguridad sobre tus finanzas?
- **Consistencia**: ¿Tus respuestas son coherentes?
- **Honestidad**: ¿Hay señales de que ocultas información?

**Ejemplo Real:**
```
Pregunta: "¿Ha tenido problemas para pagar préstamos?"
Respuesta nerviosa + pausas largas = Posible riesgo
Respuesta confiada + directa = Bajo riesgo
```

### **Para Técnicos** 🔧

**Arquitectura:**
- **Frontend**: Angular PWA (`avi_lab/src/app/`)
- **Voice Processing**: Web Audio API + Speech Analysis
- **Dataset**: 55 preguntas estructuradas en 8 categorías
- **Output**: JSON payload para HASE integration

**Tecnologías:**
```typescript
// Core service
VoiceAnalysisService {
  analyzeVoiceResponse(audioBlob: Blob, questionId: number): VoiceAnalysisResult
  generateHASEPayload(): HASEScorePayload
  calculateStressIndicators(audioBlob: Blob): StressMetrics
}
```

**Métricas Generadas:**
```json
{
  "stressIndicators": {
    "averageStress": 0.3,           // 0-1 scale
    "peakStressPoints": [12, 24, 45],
    "stressTrend": "ascending"
  },
  "keywordVerification": {
    "consistencyScore": 0.85,       // 0-1 scale
    "flagWordsDetected": ["problema", "dificultad"],
    "confidenceKeywords": ["seguro", "responsable"]
  },
  "behavioralFlags": {
    "evasiveResponses": 2,          // count
    "longPauses": 5,               // count
    "speechSpeedVariance": 0.4     // variance metric
  }
}
```

**Integration Points:**
- **Input**: Customer voice during interview
- **Output**: Structured payload to HASE
- **API**: REST endpoints for real-time analysis
- **Storage**: Session data for audit trails

---

## 📊 **2. HASE - Hyperadaptive Scoring Engine**

### **Para No Técnicos** 👥

**¿Qué es?**
Un sistema inteligente que aprende y se adapta continuamente para evaluar riesgo financiero.

**¿Cómo funciona?**
- Analiza tu comportamiento financiero histórico
- Combina datos de AVI (voz) con patrones de uso
- Se adapta en tiempo real a nuevos comportamientos del mercado
- Mejora sus predicciones con cada decisión

**¿Qué evalúa?**
- **Patrones de pago**: ¿Pagas antes, a tiempo, o tarde?
- **Uso de servicios**: ¿Cómo usas apps financieras?
- **Consistencia**: ¿Tus hábitos son estables?
- **Señales de alerta**: ¿Hay cambios preocupantes?

**Ejemplo de Adaptación:**
```
Mes 1: Cliente tipo A = Bajo riesgo
[3 meses después, muchos clientes tipo A fallan]
HASE se adapta: Cliente tipo A = Riesgo medio
[El sistema aprende automáticamente]
```

### **Para Técnicos** 🔧

**Arquitectura:**
- **Core Engine**: Python ML pipeline (`agents/hase/src/`)
- **Models**: XGBoost + Logistic Regression (`.joblib` files)
- **Features**: 50+ behavioral and transactional features
- **Adaptation**: Online learning with concept drift detection

**Algoritmos:**
```python
class HyperAdaptiveScorer:
    def __init__(self):
        self.base_model = XGBoostClassifier()
        self.adaptation_layer = OnlineLearning()
        self.concept_drift_detector = DriftDetector()

    def score_customer(self, features, avi_payload=None):
        # Base scoring
        base_score = self.base_model.predict(features)

        # AVI integration
        if avi_payload:
            voice_adjustment = self.calculate_voice_adjustment(avi_payload)
            base_score = self.adjust_score(base_score, voice_adjustment)

        # Hyperadaptive layer
        adapted_score = self.adaptation_layer.adapt(base_score)

        return adapted_score
```

**Features Engineering:**
```python
# Behavioral features
payment_consistency = calculate_payment_patterns(transactions)
usage_intensity = analyze_app_usage(sessions)
stress_indicators = extract_stress_signals(behavior_logs)

# AVI features integration
voice_trust_score = avi_payload['overallScore']['voiceTrust']
stress_risk_penalty = avi_payload['overallScore']['stressRisk']
consistency_boost = avi_payload['overallScore']['consistency']

# Final feature vector
feature_vector = combine_features(
    behavioral_features, avi_features, contextual_features
)
```

**Model Files:**
- `models/hase/hase_xgboost_model.joblib` - Primary model
- `models/hase/hase_logistic_baseline.joblib` - Fallback model
- `models/hase/hase_xgboost_metrics.json` - Performance metrics
- `models/hase/hase_logistic_metrics.json` - Baseline metrics

---

## 🎯 **3. PIA - Motor de Decisión Inteligente**

### **Para No Técnicos** 👥

**¿Qué es?**
El "cerebro" que toma las decisiones finales sobre créditos combinando toda la información.

**¿Cómo funciona?**
- Recibe el análisis de AVI (tu voz)
- Recibe la evaluación de HASE (tu comportamiento)
- Aplica las reglas de negocio de la institución
- Toma una decisión final explicable

**Tipos de Decisión:**
1. **✅ Aprobado**: Todo perfecto, condiciones normales
2. **⚠️ Aprobado con Protección**: Riesgo controlable, con salvaguardas
3. **🔍 Revisión Manual**: Señales mixtas, necesita análisis humano
4. **❌ Rechazado**: Riesgo muy alto

**Ejemplo de Lógica:**
```
AVI Score: 0.8 (bueno) + HASE Score: 0.6 (medio) =
PIA Decisión: "Aprobado con protección TIR"
```

### **Para Técnicos** 🔧

**Arquitectura:**
- **Decision Engine**: Python rule engine (`agents/pia/src/`)
- **TIR Calculator**: Financial math engine for protection scenarios
- **Integration Layer**: Combines AVI + HASE + business rules
- **Output**: Structured decision with explanations

**Core Logic:**
```python
class PIADecisionEngine:
    def make_decision(self, avi_score, hase_score, customer_profile):
        # Weight components
        voice_component = avi_score * 0.3
        behavioral_component = hase_score * 0.5
        profile_component = self.score_profile(customer_profile) * 0.2

        # Combined score
        combined_score = voice_component + behavioral_component + profile_component

        # Apply business rules
        decision = self.apply_business_rules(combined_score, customer_profile)

        # Calculate protections if approved
        if decision.approved:
            protection_scenarios = self.calculate_tir_protections(customer_profile)
            decision.protections = protection_scenarios

        return decision
```

**Decision Matrix:**
```python
DECISION_THRESHOLDS = {
    'AUTO_APPROVE': 0.8,        # Score >= 0.8
    'APPROVE_WITH_PROTECTION': 0.6,  # 0.6 <= Score < 0.8
    'MANUAL_REVIEW': 0.4,       # 0.4 <= Score < 0.6
    'AUTO_REJECT': 0.0          # Score < 0.4
}

RISK_FACTORS = {
    'HIGH_STRESS_VOICE': -0.1,   # AVI penalty
    'INCONSISTENT_BEHAVIOR': -0.15,  # HASE penalty
    'FIRST_TIME_CUSTOMER': -0.05,    # Profile penalty
}
```

**Integration Points:**
- **Input**: AVI payload + HASE score + customer data
- **Output**: Decision + explanations + protection scenarios
- **API**: FastAPI endpoints (`/pia/protection/evaluate`)

---

## 💰 **4. TIR/Protección - Motor Financiero**

### **Para No Técnicos** 👥

**¿Qué es?**
Un calculador inteligente que crea "planes B" para protegerte si tienes problemas financieros.

**¿Cómo funciona?**
- Calcula diferentes escenarios de pago
- Mantiene la rentabilidad mínima para la institución (TIR)
- Te ofrece opciones flexibles sin penalizarte

**Tipos de Protección:**
1. **Diferir Pagos**: Mueve pagos a fechas futuras
2. **Reducir Cuotas**: Pagos más bajos por más tiempo
3. **Plan Balloon**: Cuotas bajas + pago final grande
4. **Pausa Temporal**: Suspende pagos por emergencia

**Ejemplo Real:**
```
Situación: Perdiste trabajo temporalmente
Opción 1: Diferir 3 pagos (TIR: 18.5%)
Opción 2: Reducir cuota 40% por 6 meses (TIR: 18.2%)
Opción 3: Pausa 2 meses + reestructura (TIR: 18.8%)
```

### **Para Técnicos** 🔧

**Arquitectura:**
- **TIR Engine**: Financial mathematics (`agents/pia/src/tir_equilibrium_engine.py`)
- **Configuration**: Policy rules (`config/financial.yml`)
- **Scenarios**: Multiple protection algorithms
- **Validation**: IRR calculations and constraints

**Core Algorithm:**
```python
class TIREquilibriumEngine:
    def calculate_protection_scenarios(self, loan_profile):
        scenarios = []

        # Scenario 1: Defer payments
        defer_scenario = self.calculate_defer_scenario(
            loan_profile, defer_months=3
        )
        if self.validate_tir(defer_scenario):
            scenarios.append(defer_scenario)

        # Scenario 2: Reduce payments
        reduction_scenario = self.calculate_reduction_scenario(
            loan_profile, reduction_percent=0.4, extension_months=6
        )
        if self.validate_tir(reduction_scenario):
            scenarios.append(reduction_scenario)

        # Scenario 3: Balloon payment
        balloon_scenario = self.calculate_balloon_scenario(
            loan_profile, balloon_percent=0.3
        )
        if self.validate_tir(balloon_scenario):
            scenarios.append(balloon_scenario)

        return scenarios

    def validate_tir(self, scenario):
        calculated_tir = self.calculate_irr(scenario.cash_flows)
        minimum_tir = self.config['minimum_tir']
        return calculated_tir >= minimum_tir
```

**Financial Math:**
```python
def calculate_irr(cash_flows):
    """Calculate Internal Rate of Return"""
    return np.irr(cash_flows)

def calculate_npv(cash_flows, discount_rate):
    """Calculate Net Present Value"""
    return np.npv(discount_rate, cash_flows)

def generate_protection_cashflow(original_loan, protection_type):
    """Generate modified cash flow for protection scenario"""
    if protection_type == 'DEFER':
        return defer_cashflow(original_loan)
    elif protection_type == 'REDUCE':
        return reduce_cashflow(original_loan)
    elif protection_type == 'BALLOON':
        return balloon_cashflow(original_loan)
```

**Configuration:**
```yaml
# config/financial.yml
tir_policies:
  minimum_tir: 0.18           # 18% minimum
  maximum_defer_months: 6
  maximum_reduction_percent: 0.5
  balloon_threshold: 0.3

protection_rules:
  defer_eligible_score: 0.6
  reduction_eligible_score: 0.5
  balloon_eligible_score: 0.7
```

---

## 📞 **5. Agente de Postventa - Asistente Inteligente**

### **Para No Técnicos** 👥

**¿Qué es?**
Tu asistente personal 24/7 que responde preguntas y gestiona tu cuenta.

**¿Cómo funciona?**
- Entiende preguntas en lenguaje natural
- Busca información específica en documentos técnicos
- Puede procesar imágenes y audio
- Gestiona casos y seguimientos

**¿Qué puede hacer?**
- **Responder dudas**: Sobre productos, procesos, requisitos
- **Procesar documentos**: Lee PDFs, imágenes, facturas
- **Gestionar casos**: Abre tickets, da seguimiento
- **Asistir 24/7**: Disponible siempre

**Ejemplo de Interacción:**
```
Cliente: "¿Puedo diferir mi pago de enero?"
Agente: "Sí, según tu perfil puedes diferir hasta 3 pagos.
         ¿Te ayudo a iniciar el proceso?"
```

### **Para Técnicos** 🔧

**Arquitectura:**
- **RAG Engine**: Retrieval-Augmented Generation
- **Vector Store**: Pinecone for document embeddings
- **LLM**: OpenAI GPT-4 for response generation
- **Hybrid Search**: BM25 + semantic search

**Core Components:**
```python
# storage.py - Case management
class CaseManager:
    def create_case(self, customer_id, query, media_items=None):
        case = {
            'case_id': generate_case_id(),
            'customer_id': customer_id,
            'query': query,
            'status': 'OPEN',
            'created_at': datetime.now()
        }
        return self.store_case(case)

# query.py - RAG implementation
class RAGQueryEngine:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = Pinecone.from_existing_index()
        self.llm = ChatOpenAI(model="gpt-4")

    def query(self, question):
        # Hybrid retrieval
        semantic_results = self.vectorstore.similarity_search(question)
        keyword_results = self.bm25_search(question)

        # Combine and rank
        context = self.combine_results(semantic_results, keyword_results)

        # Generate response
        response = self.llm.generate(
            context=context, question=question
        )
        return response
```

**Data Processing:**
```python
# Media processing capabilities
def process_media_item(media_item):
    if media_item.type == 'image':
        # OCR + vision analysis
        text = extract_text_ocr(media_item.content)
        analysis = analyze_image_openai(media_item.content)
        return combine_text_analysis(text, analysis)

    elif media_item.type == 'audio':
        # Whisper transcription
        transcript = transcribe_whisper(media_item.content)
        return process_transcript(transcript)

    elif media_item.type == 'document':
        # PDF/document processing
        text = extract_document_text(media_item.content)
        return chunk_and_embed(text)
```

**Integration Points:**
- **Input**: Natural language queries + media
- **Processing**: RAG pipeline + case management
- **Output**: Contextual responses + case tracking
- **API**: FastAPI endpoints + webhooks

---

## 📊 **6. Dashboard - Centro de Control Visual**

### **Para No Técnicos** 👥

**¿Qué es?**
Una pantalla de control que muestra todo lo que está pasando en tiempo real.

**¿Qué muestra?**
- **Métricas en vivo**: Cuántas aplicaciones, aprobaciones, rechazos
- **Análisis AVI**: Resultados de entrevistas de voz
- **Scoring HASE**: Patrones de comportamiento
- **Decisiones PIA**: Aprobaciones y protecciones
- **Casos de Postventa**: Tickets y seguimientos

**Para Quien:**
- **Ejecutivos**: Vista general de performance
- **Analistas**: Métricas detalladas y trends
- **Operadores**: Casos individuales y alertas
- **Clientes**: Su proceso personal

### **Para Técnicos** 🔧

**Arquitectura:**
- **Frontend**: React 18 + TypeScript + Vite
- **Styling**: Styled-components + design tokens
- **Charts**: Recharts for data visualization
- **State**: React hooks + context

**Components:**
```typescript
// Core dashboard components
export interface DashboardProps {
  demoData: {
    driverStates: DriverState[];
    outcomeScenarios: OutcomeScenario[];
    planSummary: PlanSummary[];
    llmOutbox: LLMAlert[];
  }
}

// Key visualizations
const Dashboard = () => (
  <DashboardLayout>
    <PortfolioOverview data={demoData} />
    <ProtectionHeatmap scenarios={scenarios} />
    <RiskCoverageChart coverage={coverage} />
    <PaymentsTelemetry payments={payments} />
    <OutcomeTable outcomes={outcomes} />
    <AlertsList alerts={alerts} />
  </DashboardLayout>
);
```

**Data Flow:**
```typescript
// Data synchronization
const syncDashboardData = () => {
  // Sync from lab datasets
  const driverStates = readCSV('data/pia/synthetic_driver_states.csv');
  const outcomes = readCSV('data/pia/pia_outcomes_log.csv');
  const features = readCSV('data/hase/pia_outcomes_features.csv');

  // Transform for visualization
  const dashboardData = transformForDashboard({
    driverStates, outcomes, features
  });

  // Update public data
  writeJSON('dashboard/public/data/', dashboardData);
};
```

**Integration:**
- **Data Sources**: Lab datasets (CSV/JSON)
- **Sync**: `npm run sync-data` script
- **Deployment**: `npm run build` + static hosting
- **Development**: `npm run dev` on port 5173

---

## 🔄 **Integración Completa del Ecosistema**

### **Flujo de Datos**
```
Cliente → PWA Angular → AVI Analysis → HASE Scoring →
PIA Decision → TIR Protection → Dashboard Update →
Postventa Follow-up
```

### **APIs y Endpoints**
```http
# AVI Integration
POST /avi/analyze-voice
POST /avi/generate-hase-payload

# HASE Scoring
POST /hase/score-customer
POST /hase/hybrid-score

# PIA Decisions
POST /pia/protection/evaluate
POST /pia/protection/evaluate_with_summary

# TIR Calculations
POST /tir/calculate-scenarios
POST /tir/validate-protection

# Postventa Support
POST /postventa/query
POST /postventa/create-case
GET /postventa/case-status/{id}

# Dashboard Data
GET /dashboard/metrics
GET /dashboard/real-time-data
```

### **Data Pipeline**
```python
# Complete integration flow
def process_complete_application(customer_application):
    # 1. AVI Voice Analysis
    avi_session = avi_service.start_interview(customer_application.id)
    avi_results = avi_service.conduct_interview(avi_session)
    avi_payload = avi_service.generate_hase_payload(avi_results)

    # 2. HASE Scoring
    behavioral_data = hase_service.get_customer_behavior(customer_application.id)
    hybrid_score = hase_service.calculate_hybrid_score(avi_payload, behavioral_data)

    # 3. PIA Decision
    decision = pia_service.make_decision(
        avi_score=avi_payload.overall_score,
        hase_score=hybrid_score.final_score,
        customer_profile=customer_application
    )

    # 4. TIR Protection (if approved)
    if decision.approved:
        protection_scenarios = tir_service.calculate_protections(
            customer_application.loan_profile
        )
        decision.protections = protection_scenarios

    # 5. Update Dashboard
    dashboard_service.update_metrics(decision)

    # 6. Setup Postventa
    postventa_service.create_customer_profile(customer_application.id)

    return decision
```

---

Esta documentación cubre **completamente** todos los componentes del ecosistema RAG Proactive Lab, explicando tanto la funcionalidad para usuarios finales como la implementación técnica para desarrolladores.