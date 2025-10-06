import { Injectable } from '@angular/core';
import {
  AVIQuestionEnhanced,
  ALL_AVI_QUESTIONS,
  getQuestionsByCategory,
  getCriticalQuestions,
  AVICategory
} from '../../data/avi-questions-complete';
import { VoiceAnalysisResult, InterviewSession, VoiceAnalysisService } from './voice-analysis.service';

export interface MicroLocalAnalysis {
  questionId: string;
  category: AVICategory;
  localRiskScore: number;
  crossValidationScore: number;
  coherencyFlags: string[];
  recommendedFollowUp: string[];
  verificationTriggers: string[];
}

export interface LocalConsistencyCheck {
  questionPair: [string, string];
  consistencyScore: number;
  inconsistencyFlags: string[];
  suggestedInvestigation: string[];
}

export interface MicroLocalSession {
  sessionId: string;
  totalQuestions: number;
  completedQuestions: number;
  microAnalyses: MicroLocalAnalysis[];
  consistencyChecks: LocalConsistencyCheck[];
  overallCoherenceScore: number;
  riskAreas: string[];
  nextQuestionRecommendations: AVIQuestionEnhanced[];
}

@Injectable({
  providedIn: 'root'
})
export class MicroLocalAnalysisService {

  constructor(private voiceAnalysis: VoiceAnalysisService) {}

  /**
   * Performs micro-local analysis on a single question response
   */
  performMicroLocalAnalysis(
    voiceResult: VoiceAnalysisResult,
    previousResults: VoiceAnalysisResult[] = []
  ): MicroLocalAnalysis {
    const question = ALL_AVI_QUESTIONS.find(q => q.id === voiceResult.questionId);
    if (!question) {
      throw new Error(`Question ${voiceResult.questionId} not found`);
    }

    const localRiskScore = this.calculateLocalRiskScore(voiceResult, question);
    const crossValidationScore = this.performCrossValidation(voiceResult, previousResults);
    const coherencyFlags = this.detectCoherencyIssues(voiceResult, question, previousResults);
    const recommendedFollowUp = this.generateFollowUpRecommendations(question, voiceResult);

    return {
      questionId: voiceResult.questionId,
      category: question.category,
      localRiskScore,
      crossValidationScore,
      coherencyFlags,
      recommendedFollowUp,
      verificationTriggers: question.verificationTriggers
    };
  }

  /**
   * Builds a comprehensive micro-local session
   */
  buildMicroLocalSession(
    voiceResults: VoiceAnalysisResult[],
    targetQuestionCount: number = 55
  ): MicroLocalSession {
    const microAnalyses = voiceResults.map((result, index) =>
      this.performMicroLocalAnalysis(result, voiceResults.slice(0, index))
    );

    const consistencyChecks = this.performConsistencyChecks(voiceResults);
    const overallCoherenceScore = this.calculateOverallCoherence(microAnalyses, consistencyChecks);
    const riskAreas = this.identifyRiskAreas(microAnalyses);
    const nextQuestions = this.recommendNextQuestions(voiceResults, targetQuestionCount);

    return {
      sessionId: `micro_local_${Date.now()}`,
      totalQuestions: targetQuestionCount,
      completedQuestions: voiceResults.length,
      microAnalyses,
      consistencyChecks,
      overallCoherenceScore,
      riskAreas,
      nextQuestionRecommendations: nextQuestions
    };
  }

  /**
   * Validates mathematical coherence across financial questions
   */
  validateFinancialCoherence(voiceResults: VoiceAnalysisResult[]): {
    coherent: boolean;
    inconsistencies: string[];
    suggestedQuestions: AVIQuestionEnhanced[];
  } {
    const financialQuestions = [
      'ingresos_promedio_diarios',
      'gasto_diario_gasolina',
      'vueltas_por_dia',
      'pasajeros_por_vuelta',
      'tarifa_por_pasajero',
      'pago_semanal_tarjeta',
      'gastos_mordidas_cuotas'
    ];

    const relevantResults = voiceResults.filter(r =>
      financialQuestions.includes(r.questionId)
    );

    const inconsistencies: string[] = [];
    let coherent = true;

    // Mock financial coherence checks
    if (relevantResults.length >= 3) {
      // Check if income calculations make sense
      const incomeBased = relevantResults.find(r => r.questionId === 'ingresos_promedio_diarios');
      const vueltas = relevantResults.find(r => r.questionId === 'vueltas_por_dia');
      const pasajeros = relevantResults.find(r => r.questionId === 'pasajeros_por_vuelta');
      const tarifa = relevantResults.find(r => r.questionId === 'tarifa_por_pasajero');

      if (incomeBased && vueltas && pasajeros && tarifa) {
        // This would contain actual mathematical validation logic
        if (Math.random() > 0.7) { // Simulate inconsistency
          coherent = false;
          inconsistencies.push('ingresos_no_coinciden_con_operacion');
        }
      }

      // Check expense vs income ratio
      const expenses = relevantResults.filter(r =>
        ['gasto_diario_gasolina', 'pago_semanal_tarjeta', 'gastos_mordidas_cuotas'].includes(r.questionId)
      );

      if (expenses.length >= 2 && incomeBased) {
        if (Math.random() > 0.8) { // Simulate high expense ratio
          coherent = false;
          inconsistencies.push('gastos_exceden_ingresos_declarados');
        }
      }
    }

    // Suggest additional financial questions for validation
    const suggestedQuestions = this.getFinancialValidationQuestions(relevantResults);

    return {
      coherent,
      inconsistencies,
      suggestedQuestions
    };
  }

  /**
   * Generates a comprehensive micro-local report
   */
  generateMicroLocalReport(session: MicroLocalSession): string {
    const report = {
      session: {
        id: session.sessionId,
        progress: `${session.completedQuestions}/${session.totalQuestions}`,
        overallCoherence: session.overallCoherenceScore,
        riskLevel: this.determineRiskLevel(session.overallCoherenceScore, session.riskAreas)
      },
      categoryAnalysis: this.analyzeCategoryPerformance(session.microAnalyses),
      riskAreas: session.riskAreas,
      consistencyIssues: session.consistencyChecks
        .filter(check => check.consistencyScore < 0.6)
        .map(check => ({
          questionPair: check.questionPair,
          score: check.consistencyScore,
          flags: check.inconsistencyFlags
        })),
      recommendations: {
        nextQuestions: session.nextQuestionRecommendations.slice(0, 5).map(q => ({
          id: q.id,
          question: q.question,
          priority: q.weight,
          category: q.category
        })),
        investigationAreas: this.getInvestigationAreas(session.microAnalyses),
        urgentFlags: session.microAnalyses
          .filter(analysis => analysis.localRiskScore < 0.4)
          .map(analysis => analysis.questionId)
      },
      summary: {
        totalRiskFlags: session.microAnalyses.reduce((acc, a) => acc + a.coherencyFlags.length, 0),
        avgLocalRiskScore: session.microAnalyses.reduce((acc, a) => acc + a.localRiskScore, 0) / session.microAnalyses.length,
        criticalQuestionsCompleted: session.microAnalyses
          .filter(a => {
            const q = ALL_AVI_QUESTIONS.find(q => q.id === a.questionId);
            return q && q.weight >= 9;
          }).length
      }
    };

    return JSON.stringify(report, null, 2);
  }

  // Private helper methods

  private calculateLocalRiskScore(
    voiceResult: VoiceAnalysisResult,
    question: AVIQuestionEnhanced
  ): number {
    let riskScore = voiceResult.voiceScore; // Start with voice score

    // Adjust based on question criticality
    const weightFactor = question.weight / 10.0;
    const stressFactor = question.stressLevel / 5.0;

    // Higher weight questions with low scores are riskier
    if (question.weight >= 9 && voiceResult.voiceScore < 0.6) {
      riskScore *= 0.7; // Significant penalty
    }

    // Multiple stress indicators increase risk
    if (voiceResult.stressIndicators.length >= 2) {
      riskScore *= (1.0 - stressFactor * 0.2);
    }

    // Risk flags compound the risk
    const flagPenalty = voiceResult.riskFlags.length * 0.1;
    riskScore = Math.max(0, riskScore - flagPenalty);

    return Math.max(0, Math.min(1, riskScore));
  }

  private performCrossValidation(
    currentResult: VoiceAnalysisResult,
    previousResults: VoiceAnalysisResult[]
  ): number {
    if (previousResults.length === 0) return 0.8; // Neutral score for first question

    const currentQuestion = ALL_AVI_QUESTIONS.find(q => q.id === currentResult.questionId);
    if (!currentQuestion) return 0.5;

    // Find related questions in the same category or with verification triggers
    const relatedResults = previousResults.filter(prevResult => {
      const prevQuestion = ALL_AVI_QUESTIONS.find(q => q.id === prevResult.questionId);
      if (!prevQuestion) return false;

      return prevQuestion.category === currentQuestion.category ||
             currentQuestion.verificationTriggers.some(trigger =>
               prevQuestion.verificationTriggers.includes(trigger)
             );
    });

    if (relatedResults.length === 0) return 0.8;

    // Calculate consistency based on similar response patterns
    let consistencyScore = 0.8;
    const avgRelatedScore = relatedResults.reduce((acc, r) => acc + r.voiceScore, 0) / relatedResults.length;

    // If current score deviates significantly from related scores, reduce consistency
    const deviation = Math.abs(currentResult.voiceScore - avgRelatedScore);
    consistencyScore -= deviation * 0.5;

    // Check for common stress indicators
    const commonStressCount = currentResult.stressIndicators.filter(indicator =>
      relatedResults.some(r => r.stressIndicators.includes(indicator))
    ).length;

    if (commonStressCount >= 2) {
      consistencyScore *= 0.9; // Slight penalty for consistent stress patterns
    }

    return Math.max(0.2, Math.min(1, consistencyScore));
  }

  private detectCoherencyIssues(
    voiceResult: VoiceAnalysisResult,
    question: AVIQuestionEnhanced,
    previousResults: VoiceAnalysisResult[]
  ): string[] {
    const flags: string[] = [];

    // Check for dramatic score changes
    if (previousResults.length > 0) {
      const recentAvg = previousResults.slice(-3).reduce((acc, r) => acc + r.voiceScore, 0) / Math.min(3, previousResults.length);
      const scoreDiff = Math.abs(voiceResult.voiceScore - recentAvg);

      if (scoreDiff > 0.4) {
        flags.push('cambio_drastico_puntuacion');
      }
    }

    // Check for inconsistent stress patterns
    if (question.stressLevel <= 2 && voiceResult.stressIndicators.length >= 2) {
      flags.push('estres_inesperado_pregunta_simple');
    }

    if (question.stressLevel >= 4 && voiceResult.stressIndicators.length === 0) {
      flags.push('calma_sospechosa_pregunta_estresante');
    }

    // Check response time coherence
    const expectedTime = question.analytics.expectedResponseTime;
    if (voiceResult.responseTime > expectedTime * 3) {
      flags.push('tiempo_respuesta_excesivo');
    } else if (voiceResult.responseTime < expectedTime * 0.2) {
      flags.push('respuesta_demasiado_rapida_sospechosa');
    }

    return flags;
  }

  private generateFollowUpRecommendations(
    question: AVIQuestionEnhanced,
    voiceResult: VoiceAnalysisResult
  ): string[] {
    const recommendations: string[] = [];

    // Add question's predefined follow-up questions
    if (question.followUpQuestions) {
      recommendations.push(...question.followUpQuestions);
    }

    // Add dynamic recommendations based on analysis
    if (voiceResult.riskFlags.includes('score_bajo')) {
      recommendations.push('Profundizar en las razones de la respuesta evasiva');
      recommendations.push('Verificar información con fuentes externas');
    }

    if (voiceResult.stressIndicators.length >= 2) {
      recommendations.push('Repetir pregunta de manera más casual');
      recommendations.push('Hacer preguntas de verificación cruzada');
    }

    if (question.weight >= 9 && voiceResult.voiceScore < 0.6) {
      recommendations.push('Esta es una pregunta crítica - considerar re-evaluación');
      recommendations.push('Solicitar documentación de respaldo');
    }

    return recommendations;
  }

  private performConsistencyChecks(voiceResults: VoiceAnalysisResult[]): LocalConsistencyCheck[] {
    const checks: LocalConsistencyCheck[] = [];

    // Define question pairs that should be consistent
    const consistencyPairs = [
      ['ingresos_promedio_diarios', 'vueltas_por_dia'],
      ['gasto_diario_gasolina', 'vueltas_por_tanque'],
      ['edad', 'anos_en_ruta'],
      ['tipo_operacion', 'valor_unidad_transporte'],
      ['creditos_anteriores', 'problemas_pagos']
    ];

    consistencyPairs.forEach(pair => {
      const result1 = voiceResults.find(r => r.questionId === pair[0]);
      const result2 = voiceResults.find(r => r.questionId === pair[1]);

      if (result1 && result2) {
        const consistencyScore = this.calculatePairConsistency(result1, result2);
        const flags = this.generateConsistencyFlags(result1, result2, consistencyScore);
        const investigation = this.suggestInvestigation(pair, consistencyScore, flags);

        checks.push({
          questionPair: pair as [string, string],
          consistencyScore,
          inconsistencyFlags: flags,
          suggestedInvestigation: investigation
        });
      }
    });

    return checks;
  }

  private calculatePairConsistency(result1: VoiceAnalysisResult, result2: VoiceAnalysisResult): number {
    // Base consistency on score similarity and stress indicator patterns
    const scoreDiff = Math.abs(result1.voiceScore - result2.voiceScore);
    const baseConsistency = 1.0 - (scoreDiff * 0.8);

    // Check for common stress indicators (should be consistent for related questions)
    const commonStress = result1.stressIndicators.filter(s => result2.stressIndicators.includes(s));
    const stressBonus = commonStress.length * 0.1;

    return Math.min(1, Math.max(0.2, baseConsistency + stressBonus));
  }

  private generateConsistencyFlags(
    result1: VoiceAnalysisResult,
    result2: VoiceAnalysisResult,
    consistencyScore: number
  ): string[] {
    const flags: string[] = [];

    if (consistencyScore < 0.4) {
      flags.push('inconsistencia_grave');
    } else if (consistencyScore < 0.6) {
      flags.push('inconsistencia_moderada');
    }

    // Check for contradictory stress patterns
    const highStress1 = result1.stressIndicators.length >= 2;
    const highStress2 = result2.stressIndicators.length >= 2;

    if (highStress1 !== highStress2) {
      flags.push('patron_estres_contradictorio');
    }

    return flags;
  }

  private suggestInvestigation(
    pair: string[],
    consistencyScore: number,
    flags: string[]
  ): string[] {
    const suggestions: string[] = [];

    if (consistencyScore < 0.5) {
      suggestions.push(`Investigar discrepancia entre ${pair[0]} y ${pair[1]}`);
      suggestions.push('Solicitar aclaración o documentación de respaldo');
    }

    if (flags.includes('patron_estres_contradictorio')) {
      suggestions.push('Repetir preguntas en orden diferente para verificar consistencia');
    }

    return suggestions;
  }

  private calculateOverallCoherence(
    analyses: MicroLocalAnalysis[],
    consistencyChecks: LocalConsistencyCheck[]
  ): number {
    if (analyses.length === 0) return 0;

    const avgLocalRisk = analyses.reduce((acc, a) => acc + a.localRiskScore, 0) / analyses.length;
    const avgCrossValidation = analyses.reduce((acc, a) => acc + a.crossValidationScore, 0) / analyses.length;

    let overallScore = (avgLocalRisk + avgCrossValidation) / 2;

    // Penalize based on consistency check failures
    const failedChecks = consistencyChecks.filter(c => c.consistencyScore < 0.6).length;
    const checkPenalty = failedChecks * 0.1;

    overallScore = Math.max(0, overallScore - checkPenalty);

    return Math.min(1, overallScore);
  }

  private identifyRiskAreas(analyses: MicroLocalAnalysis[]): string[] {
    const riskAreas: string[] = [];
    const categoryRisks: Record<string, number[]> = {};

    // Group analyses by category and calculate average risk
    analyses.forEach(analysis => {
      const category = analysis.category;
      if (!categoryRisks[category]) {
        categoryRisks[category] = [];
      }
      categoryRisks[category].push(analysis.localRiskScore);
    });

    // Identify categories with high risk (low scores)
    Object.entries(categoryRisks).forEach(([category, scores]) => {
      const avgRisk = scores.reduce((acc, s) => acc + s, 0) / scores.length;
      if (avgRisk < 0.5) {
        riskAreas.push(category);
      }
    });

    return riskAreas;
  }

  private recommendNextQuestions(
    completedResults: VoiceAnalysisResult[],
    targetTotal: number
  ): AVIQuestionEnhanced[] {
    const completedIds = completedResults.map(r => r.questionId);
    const remaining = ALL_AVI_QUESTIONS.filter(q => !completedIds.includes(q.id));

    // Prioritize based on risk areas and missing critical questions
    const critical = remaining.filter(q => q.weight >= 9);
    const highStress = remaining.filter(q => q.stressLevel >= 4);

    // Combine and deduplicate
    const prioritized = [...new Set([...critical, ...highStress])];
    const others = remaining.filter(q => !prioritized.includes(q));

    return [...prioritized, ...others].slice(0, Math.min(10, targetTotal - completedResults.length));
  }

  private analyzeCategoryPerformance(analyses: MicroLocalAnalysis[]): Record<string, any> {
    const performance: Record<string, any> = {};

    Object.values(AVICategory).forEach(category => {
      const categoryAnalyses = analyses.filter(a => a.category === category);
      if (categoryAnalyses.length > 0) {
        performance[category] = {
          questionsCompleted: categoryAnalyses.length,
          avgRiskScore: categoryAnalyses.reduce((acc, a) => acc + a.localRiskScore, 0) / categoryAnalyses.length,
          totalFlags: categoryAnalyses.reduce((acc, a) => acc + a.coherencyFlags.length, 0),
          riskLevel: this.categorizeRiskLevel(
            categoryAnalyses.reduce((acc, a) => acc + a.localRiskScore, 0) / categoryAnalyses.length
          )
        };
      }
    });

    return performance;
  }

  private getInvestigationAreas(analyses: MicroLocalAnalysis[]): string[] {
    const areas = new Set<string>();

    analyses.forEach(analysis => {
      if (analysis.localRiskScore < 0.5) {
        areas.add(analysis.category);
      }
      analysis.coherencyFlags.forEach(flag => {
        if (flag.includes('inconsistencia') || flag.includes('sospechosa')) {
          areas.add(`investigar_${analysis.questionId}`);
        }
      });
    });

    return Array.from(areas);
  }

  private getFinancialValidationQuestions(completedResults: VoiceAnalysisResult[]): AVIQuestionEnhanced[] {
    const completedIds = completedResults.map(r => r.questionId);
    const financialValidationIds = [
      'coherencia_ingresos_gastos',
      'confirmacion_datos_criticos',
      'compromisos_existentes',
      'ahorros_emergencia'
    ];

    return ALL_AVI_QUESTIONS.filter(q =>
      financialValidationIds.includes(q.id) && !completedIds.includes(q.id)
    );
  }

  private determineRiskLevel(coherenceScore: number, riskAreas: string[]): string {
    if (coherenceScore < 0.4 || riskAreas.length >= 3) return 'HIGH';
    if (coherenceScore < 0.6 || riskAreas.length >= 2) return 'MEDIUM';
    return 'LOW';
  }

  private categorizeRiskLevel(avgRiskScore: number): string {
    if (avgRiskScore < 0.4) return 'HIGH';
    if (avgRiskScore < 0.6) return 'MEDIUM';
    return 'LOW';
  }
}