import { Component, inject, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { AudioRecorderService } from '../services/audio-recorder.service';
import { VoiceApiService } from '../services/voice-api.service';
import { VoiceEvaluateRes } from '../core/avi-api-contracts';
import {
  ALL_AVI_QUESTIONS,
  AVI_CONFIG,
  getQuestionsByCategory,
  getCriticalQuestions,
  getQuestionsForDemo
} from '../../data/avi-questions-complete';
import { AVICategory, AVIQuestionEnhanced } from '../../data/avi-types';

@Component({
  standalone: true,
  selector: 'app-runner',
  imports: [CommonModule],
  template: `
  <div class="card">
    <h2>AVI Runner - Laboratorio de Voz</h2>
    <p class="muted">{{ aviConfig.system_status }}</p>
    <p class="muted">{{ aviConfig.total_questions }} preguntas | {{ aviConfig.critical_questions }} críticas | {{ aviConfig.high_stress_questions }} alto estrés</p>

    <!-- Question Selection -->
    <div class="question-section" style="margin: 16px 0;">
      <h3>Selección de Pregunta</h3>
      <div class="row" style="margin-bottom: 12px;">
        <select (change)="onQuestionModeChange($event)" style="margin-right: 8px;">
          <option value="demo">Demo (5 preguntas)</option>
          <option value="critical">Solo Críticas (peso ≥9)</option>
          <option value="high-stress">Alto Estrés (nivel ≥4)</option>
          <option value="category">Por Categoría</option>
          <option value="all">Todas (55 preguntas)</option>
        </select>

        <select *ngIf="questionMode() === 'category'" (change)="onCategoryChange($event)" style="margin-right: 8px;">
          <option value="">Selecciona Categoría</option>
          <option value="basic_info">Info Básica ({{ getCategoryCount('basic_info') }})</option>
          <option value="daily_operation">Operación Diaria ({{ getCategoryCount('daily_operation') }})</option>
          <option value="operational_costs">Costos Operativos ({{ getCategoryCount('operational_costs') }})</option>
          <option value="business_structure">Estructura Empresarial ({{ getCategoryCount('business_structure') }})</option>
          <option value="assets_patrimony">Activos y Patrimonio ({{ getCategoryCount('assets_patrimony') }})</option>
          <option value="credit_history">Historial Crediticio ({{ getCategoryCount('credit_history') }})</option>
          <option value="payment_intention">Intención de Pago ({{ getCategoryCount('payment_intention') }})</option>
          <option value="risk_evaluation">Evaluación de Riesgo ({{ getCategoryCount('risk_evaluation') }})</option>
        </select>
      </div>

      <div class="available-questions" style="max-height: 200px; overflow-y: auto; border: 1px solid #ddd; padding: 8px;">
        <p style="font-weight: bold;">{{ filteredQuestions().length }} preguntas disponibles:</p>
        <div *ngFor="let q of filteredQuestions(); let i = index"
             [class]="'question-item ' + (selectedQuestion()?.id === q.id ? 'selected' : '')"
             (click)="selectQuestion(q)"
             style="cursor: pointer; padding: 4px; margin: 2px 0; border-radius: 4px; border: 1px solid transparent;">
          <strong>{{ i + 1 }}.</strong> {{ q.question }}
          <span class="question-meta" style="font-size: 0.8em; color: #666;">
            (Peso: {{ q.weight }}, Estrés: {{ q.stressLevel }}, {{ q.category }})
          </span>
        </div>
      </div>

      <div *ngIf="selectedQuestion()" class="selected-question" style="margin-top: 12px; padding: 12px; background: #f0f8ff; border-radius: 4px;">
        <h4>Pregunta Seleccionada:</h4>
        <p><strong>{{ selectedQuestion()!.question }}</strong></p>
        <div class="question-details" style="font-size: 0.9em; color: #555;">
          <p><strong>Peso:</strong> {{ selectedQuestion()!.weight }}/10 |
             <strong>Estrés:</strong> {{ selectedQuestion()!.stressLevel }}/5 |
             <strong>Impacto:</strong> {{ selectedQuestion()!.riskImpact }}</p>
          <p><strong>Tiempo estimado:</strong> {{ selectedQuestion()!.estimatedTime }}s |
             <strong>Categoría:</strong> {{ selectedQuestion()!.category }}</p>
          <div *ngIf="selectedQuestion()!.followUpQuestions && selectedQuestion()!.followUpQuestions!.length > 0">
            <strong>Preguntas de seguimiento:</strong>
            <ul>
              <li *ngFor="let followUp of selectedQuestion()!.followUpQuestions">{{ followUp }}</li>
            </ul>
          </div>
        </div>
      </div>
    </div>

    <!-- Audio Recording -->
    <div class="recording-section">
      <h3>Grabación de Audio</h3>
      <div class="row" style="margin: 8px 0 16px 0">
        <button class="btn btn-primary" (click)="start()" [disabled]="recording() || !selectedQuestion()">
          {{ recording() ? 'Grabando...' : 'Grabar' }}
        </button>
        <button class="btn btn-ghost" (click)="stop()" [disabled]="!recording()">Detener</button>
        <button class="btn btn-ghost" (click)="evaluate()" [disabled]="!blob() || !selectedQuestion()">Evaluar</button>
        <button class="btn btn-secondary" (click)="nextQuestion()" [disabled]="filteredQuestions().length <= 1">
          Siguiente Pregunta
        </button>
      </div>

      <div *ngIf="!selectedQuestion()" class="warning" style="color: var(--warning); font-style: italic;">
        ⚠️ Selecciona una pregunta antes de grabar
      </div>
    </div>

    <!-- Results -->
    <div *ngIf="res()" class="results-section" style="margin-top: 20px;">
      <h3>Resultados del Análisis</h3>
      <div class="row">
        <div class="col">
          <div class="score" style="font-size: 1.5em; font-weight: bold;">
            Score: {{ res()!.voiceScore | number:'1.2-2' }}
          </div>
          <div>Decision: <span class="pill">{{ res()!.decision }}</span></div>
          <div class="muted">Flags: {{ res()!.flags.join(', ') || '—' }}</div>
          <div class="muted">Pregunta ID: {{ res()!.questionId || 'N/A' }}</div>
        </div>
        <div class="col">
          <div class="muted">latencyIndex: {{ res()!.latencyIndex | number:'1.2-2' }}</div>
          <div class="muted">pitchVar: {{ res()!.pitchVar | number:'1.2-2' }}</div>
          <div class="muted">energyStability: {{ res()!.energyStability | number:'1.2-2' }}</div>
          <div class="muted">disfluencyRate: {{ res()!.disfluencyRate | number:'1.2-2' }}</div>
          <div class="muted">honestyLexicon: {{ res()!.honestyLexicon | number:'1.2-2' }}</div>
        </div>
      </div>

      <div *ngIf="res()!.transcript" style="margin-top: 12px;">
        <strong>Transcripción:</strong>
        <p style="font-style: italic; background: #f9f9f9; padding: 8px; border-radius: 4px;">
          "{{ res()!.transcript }}"
        </p>
      </div>

      <div *ngIf="res()!.words && res()!.words.length > 0" style="margin-top: 8px;">
        <strong>Palabras detectadas:</strong> {{ res()!.words.join(', ') }}
      </div>
    </div>

    <div *ngIf="error()" style="color: var(--danger); margin-top: 12px;">{{ error() }}</div>
  </div>

  <style>
    .question-item.selected {
      background-color: #e3f2fd;
      border-color: #2196f3;
    }
    .question-item:hover {
      background-color: #f5f5f5;
    }
    .question-meta {
      display: block;
      margin-left: 16px;
    }
    .pill {
      background: #2196f3;
      color: white;
      padding: 2px 8px;
      border-radius: 12px;
      font-size: 0.8em;
    }
    .score {
      color: #4caf50;
      font-weight: bold;
    }
  </style>
  `
})
export class RunnerComponent {
  private rec = inject(AudioRecorderService);
  private api = inject(VoiceApiService);

  // Existing signals
  recording = signal(false);
  blob = signal<Blob | null>(null);
  res = signal<VoiceEvaluateRes | null>(null);
  error = signal<string | null>(null);

  // New signals for questions
  questionMode = signal<string>('demo');
  selectedCategory = signal<string>('');
  filteredQuestions = signal<AVIQuestionEnhanced[]>(getQuestionsForDemo());
  selectedQuestion = signal<AVIQuestionEnhanced | null>(null);

  // Dataset configuration
  aviConfig = AVI_CONFIG;

  constructor() {
    // Initialize with first demo question
    this.selectQuestion(getQuestionsForDemo()[0]);
  }

  // Question management methods
  onQuestionModeChange(event: Event) {
    const mode = (event.target as HTMLSelectElement).value;
    this.questionMode.set(mode);
    this.selectedCategory.set('');
    this.updateFilteredQuestions();
  }

  onCategoryChange(event: Event) {
    const category = (event.target as HTMLSelectElement).value;
    this.selectedCategory.set(category);
    this.updateFilteredQuestions();
  }

  updateFilteredQuestions() {
    let questions: AVIQuestionEnhanced[] = [];

    switch (this.questionMode()) {
      case 'demo':
        questions = getQuestionsForDemo();
        break;
      case 'critical':
        questions = getCriticalQuestions();
        break;
      case 'high-stress':
        questions = ALL_AVI_QUESTIONS.filter(q => q.stressLevel >= 4);
        break;
      case 'category':
        if (this.selectedCategory()) {
          questions = getQuestionsByCategory(this.selectedCategory() as AVICategory);
        } else {
          questions = [];
        }
        break;
      case 'all':
        questions = ALL_AVI_QUESTIONS;
        break;
      default:
        questions = getQuestionsForDemo();
    }

    this.filteredQuestions.set(questions);

    // Auto-select first question if none selected or current selection not in filtered
    if (!this.selectedQuestion() ||
        !questions.find(q => q.id === this.selectedQuestion()?.id)) {
      this.selectQuestion(questions[0] || null);
    }
  }

  selectQuestion(question: AVIQuestionEnhanced | null) {
    this.selectedQuestion.set(question);
    // Clear previous results when changing questions
    this.res.set(null);
    this.error.set(null);
    this.blob.set(null);
  }

  nextQuestion() {
    const currentQuestions = this.filteredQuestions();
    const currentIndex = currentQuestions.findIndex(q => q.id === this.selectedQuestion()?.id);
    const nextIndex = (currentIndex + 1) % currentQuestions.length;
    this.selectQuestion(currentQuestions[nextIndex]);
  }

  getCategoryCount(category: string): number {
    return AVI_CONFIG.questions_by_category[category.toUpperCase()] || 0;
  }

  // Audio recording methods
  async start() {
    if (!this.selectedQuestion()) {
      this.error.set('Selecciona una pregunta primero');
      return;
    }

    this.error.set(null);
    this.res.set(null);
    this.blob.set(null);
    try {
      await this.rec.start();
      this.recording.set(true);
    } catch (e: any) {
      this.error.set('No se pudo iniciar la grabación');
    }
  }

  async stop() {
    const b = await this.rec.stop();
    this.recording.set(false);
    this.blob.set(b);
  }

  async evaluate() {
    if (!this.blob() || !this.selectedQuestion()) return;

    this.error.set(null);
    this.res.set(null);

    const questionId = this.selectedQuestion()!.id;
    const contextId = `${this.selectedQuestion()!.category}_${Date.now()}`;

    this.api.evaluateAudio(this.blob()!, questionId, contextId).subscribe({
      next: r => {
        // Enhance response with question context
        const enhancedResult = {
          ...r,
          questionId: questionId,
          contextId: contextId,
          questionText: this.selectedQuestion()!.question,
          questionWeight: this.selectedQuestion()!.weight,
          questionStressLevel: this.selectedQuestion()!.stressLevel
        };
        this.res.set(enhancedResult as any);
      },
      error: err => {
        console.error('Evaluation error:', err);
        this.error.set(`Error al evaluar pregunta "${this.selectedQuestion()!.question}" (usa Mock ON para simular)`);
      }
    });
  }
}

