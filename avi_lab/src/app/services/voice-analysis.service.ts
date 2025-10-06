import { Injectable } from '@angular/core';
import {
  AVIQuestionEnhanced,
  ALL_AVI_QUESTIONS,
  getQuestionById,
  AVICategory
} from '../../data/avi-questions-complete';

export interface VoiceAnalysisResult {
  questionId: string;
  voiceScore: number;
  stressIndicators: string[];
  truthVerificationKeywords: string[];
  riskFlags: string[];
  responseTime: number;
  analysisMetrics: {
    latencyIndex: number;
    pitchVariability: number;
    energyStability: number;
    disfluencyRate: number;
    honestyLexicon: number;
  };
}

export interface InterviewSession {
  sessionId: string;
  startTime: Date;
  questions: AVIQuestionEnhanced[];
  responses: VoiceAnalysisResult[];
  overallScore: number;
  riskLevel: 'LOW' | 'MEDIUM' | 'HIGH';
  flags: string[];
}

@Injectable({
  providedIn: 'root'
})
export class VoiceAnalysisService {

  /**
   * Analyzes voice response for a specific AVI question
   */
  analyzeVoiceResponse(
    questionId: string,
    audioData: Blob,
    transcript?: string,
    responseTime?: number
  ): VoiceAnalysisResult {
    const question = getQuestionById(questionId);
    if (!question) {
      throw new Error(`Question with id ${questionId} not found`);
    }

    // Simulate analysis based on question parameters
    const analysisMetrics = this.calculateAnalysisMetrics(question, audioData, transcript, responseTime);
    const stressIndicators = this.detectStressIndicators(question, transcript, responseTime);
    const truthKeywords = this.findTruthVerificationKeywords(question, transcript);
    const voiceScore = this.calculateVoiceScore(question, analysisMetrics, stressIndicators, truthKeywords);
    const riskFlags = this.generateRiskFlags(question, voiceScore, stressIndicators);

    return {
      questionId,
      voiceScore,
      stressIndicators,
      truthVerificationKeywords: truthKeywords,
      riskFlags,
      responseTime: responseTime || 0,
      analysisMetrics
    };
  }

  /**
   * Analyzes a complete interview session
   */
  analyzeInterviewSession(responses: VoiceAnalysisResult[]): InterviewSession {
    const overallScore = this.calculateOverallScore(responses);
    const riskLevel = this.determineRiskLevel(responses);
    const flags = this.consolidateFlags(responses);

    return {
      sessionId: this.generateSessionId(),
      startTime: new Date(),
      questions: responses.map(r => getQuestionById(r.questionId)!),
      responses,
      overallScore,
      riskLevel,
      flags
    };
  }

  /**
   * Gets recommended next questions based on current responses
   */
  getRecommendedNextQuestions(
    currentResponses: VoiceAnalysisResult[],
    category?: AVICategory,
    count: number = 5
  ): AVIQuestionEnhanced[] {
    const askedQuestionIds = currentResponses.map(r => r.questionId);
    const availableQuestions = ALL_AVI_QUESTIONS.filter(q => !askedQuestionIds.includes(q.id));

    // Prioritize high-risk questions if previous responses show concerning patterns
    const hasHighRisk = currentResponses.some(r => r.voiceScore < 0.6 || r.riskFlags.length > 2);

    let prioritizedQuestions = availableQuestions;

    if (hasHighRisk) {
      // Focus on critical and high-stress questions
      prioritizedQuestions = availableQuestions.filter(q => q.weight >= 8 || q.stressLevel >= 4);
    }

    if (category) {
      prioritizedQuestions = prioritizedQuestions.filter(q => q.category === category);
    }

    // Sort by weight (descending) and take the requested count
    return prioritizedQuestions
      .sort((a, b) => b.weight - a.weight)
      .slice(0, count);
  }

  /**
   * Exports interview results for external analysis
   */
  exportInterviewResults(session: InterviewSession): string {
    const exportData = {
      session: {
        id: session.sessionId,
        timestamp: session.startTime.toISOString(),
        overallScore: session.overallScore,
        riskLevel: session.riskLevel,
        totalQuestions: session.questions.length,
        flags: session.flags
      },
      responses: session.responses.map(r => ({
        questionId: r.questionId,
        question: getQuestionById(r.questionId)?.question,
        category: getQuestionById(r.questionId)?.category,
        weight: getQuestionById(r.questionId)?.weight,
        stressLevel: getQuestionById(r.questionId)?.stressLevel,
        voiceScore: r.voiceScore,
        responseTime: r.responseTime,
        stressIndicators: r.stressIndicators,
        riskFlags: r.riskFlags,
        analysisMetrics: r.analysisMetrics
      })),
      summary: {
        averageScore: session.responses.reduce((acc, r) => acc + r.voiceScore, 0) / session.responses.length,
        totalStressIndicators: session.responses.reduce((acc, r) => acc + r.stressIndicators.length, 0),
        totalRiskFlags: session.responses.reduce((acc, r) => acc + r.riskFlags.length, 0),
        categoryCoverage: this.getCategoryCoverage(session.questions),
        criticalQuestionsAsked: session.questions.filter(q => q.weight >= 9).length
      }
    };

    return JSON.stringify(exportData, null, 2);
  }

  // Private helper methods

  private calculateAnalysisMetrics(
    question: AVIQuestionEnhanced,
    audioData: Blob,
    transcript?: string,
    responseTime?: number
  ) {
    // Mock calculation - in real implementation, this would use actual audio analysis
    const baseLatency = responseTime ? Math.max(0, (responseTime - question.analytics.expectedResponseTime) / 1000) : 0;

    return {
      latencyIndex: Math.min(1.0, baseLatency / 5.0), // Normalize to 0-1
      pitchVariability: Math.random() * 0.3 + (question.stressLevel * 0.1), // Higher stress = more variation
      energyStability: Math.max(0.2, 1.0 - (question.stressLevel * 0.15)),
      disfluencyRate: Math.random() * 0.2 + (question.weight > 8 ? 0.1 : 0),
      honestyLexicon: this.calculateHonestyScore(transcript, question)
    };
  }

  private calculateHonestyScore(transcript?: string, question?: AVIQuestionEnhanced): number {
    if (!transcript || !question) return 0.5;

    const truthKeywords = question.analytics.truthVerificationKeywords;
    const evasiveWords = ['no_se', 'tal_vez', 'creo_que', 'posiblemente', 'quizas'];

    let honestyScore = 0.7; // Base score

    // Check for truth verification keywords (positive indicators)
    const truthMatches = truthKeywords.filter(keyword =>
      transcript.toLowerCase().includes(keyword.replace('_', ' '))
    ).length;
    honestyScore += truthMatches * 0.1;

    // Check for evasive language (negative indicators)
    const evasiveMatches = evasiveWords.filter(word =>
      transcript.toLowerCase().includes(word.replace('_', ' '))
    ).length;
    honestyScore -= evasiveMatches * 0.1;

    return Math.max(0, Math.min(1, honestyScore));
  }

  private detectStressIndicators(
    question: AVIQuestionEnhanced,
    transcript?: string,
    responseTime?: number
  ): string[] {
    const indicators: string[] = [];
    const expectedTime = question.analytics.expectedResponseTime;

    // Response time indicators
    if (responseTime) {
      if (responseTime > expectedTime * 2) {
        indicators.push('respuesta_muy_lenta');
      } else if (responseTime > expectedTime * 1.5) {
        indicators.push('demora_respuesta');
      } else if (responseTime < expectedTime * 0.3) {
        indicators.push('respuesta_demasiado_rapida');
      }
    }

    // Check question-specific stress indicators
    if (transcript) {
      question.analytics.stressIndicators.forEach(indicator => {
        const checkPhrase = indicator.replace('_', ' ');
        if (transcript.toLowerCase().includes(checkPhrase)) {
          indicators.push(indicator);
        }
      });
    }

    // Add random stress indicators for high-stress questions (simulation)
    if (question.stressLevel >= 4 && Math.random() > 0.6) {
      indicators.push('nerviosismo_detectado');
    }

    return indicators;
  }

  private findTruthVerificationKeywords(question: AVIQuestionEnhanced, transcript?: string): string[] {
    if (!transcript) return [];

    return question.analytics.truthVerificationKeywords.filter(keyword =>
      transcript.toLowerCase().includes(keyword.replace('_', ' '))
    );
  }

  private calculateVoiceScore(
    question: AVIQuestionEnhanced,
    metrics: any,
    stressIndicators: string[],
    truthKeywords: string[]
  ): number {
    let score = 0.8; // Base score

    // Weight-based adjustment
    const weightFactor = question.weight / 10.0;

    // Stress indicators penalty
    score -= stressIndicators.length * 0.1;

    // Truth keywords bonus
    score += truthKeywords.length * 0.05;

    // Metrics adjustments
    score -= metrics.latencyIndex * 0.2;
    score -= metrics.disfluencyRate * 0.3;
    score += metrics.honestyLexicon * 0.2;
    score -= metrics.pitchVariability * 0.1;
    score += metrics.energyStability * 0.1;

    // Apply weight factor
    score *= (1.0 - weightFactor * 0.2); // Higher weight questions are harder to score well

    return Math.max(0, Math.min(1, score));
  }

  private generateRiskFlags(
    question: AVIQuestionEnhanced,
    voiceScore: number,
    stressIndicators: string[]
  ): string[] {
    const flags: string[] = [];

    if (voiceScore < 0.3) flags.push('score_muy_bajo');
    if (voiceScore < 0.5) flags.push('score_bajo');

    if (stressIndicators.length >= 3) flags.push('multiples_indicadores_estres');
    if (stressIndicators.includes('nerviosismo_extremo')) flags.push('nerviosismo_extremo');
    if (stressIndicators.includes('evasion_total')) flags.push('evasion_detectada');

    // Question-specific risk flags
    if (question.weight >= 9 && voiceScore < 0.6) {
      flags.push('pregunta_critica_puntuacion_baja');
    }

    if (question.stressLevel >= 4 && stressIndicators.length >= 2) {
      flags.push('alta_tension_pregunta_estresante');
    }

    return flags;
  }

  private calculateOverallScore(responses: VoiceAnalysisResult[]): number {
    if (responses.length === 0) return 0;

    // Weighted average based on question weights
    let totalWeightedScore = 0;
    let totalWeight = 0;

    responses.forEach(response => {
      const question = getQuestionById(response.questionId);
      if (question) {
        totalWeightedScore += response.voiceScore * question.weight;
        totalWeight += question.weight;
      }
    });

    return totalWeight > 0 ? totalWeightedScore / totalWeight : 0;
  }

  private determineRiskLevel(responses: VoiceAnalysisResult[]): 'LOW' | 'MEDIUM' | 'HIGH' {
    const overallScore = this.calculateOverallScore(responses);
    const totalRiskFlags = responses.reduce((acc, r) => acc + r.riskFlags.length, 0);
    const avgRiskFlags = totalRiskFlags / responses.length;

    if (overallScore < 0.4 || avgRiskFlags >= 2) return 'HIGH';
    if (overallScore < 0.6 || avgRiskFlags >= 1) return 'MEDIUM';
    return 'LOW';
  }

  private consolidateFlags(responses: VoiceAnalysisResult[]): string[] {
    const allFlags = responses.flatMap(r => r.riskFlags);
    return Array.from(new Set(allFlags)); // Remove duplicates
  }

  private generateSessionId(): string {
    return `avi_session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  private getCategoryCoverage(questions: AVIQuestionEnhanced[]): Record<string, number> {
    const coverage: Record<string, number> = {};

    Object.values(AVICategory).forEach(category => {
      coverage[category] = questions.filter(q => q.category === category).length;
    });

    return coverage;
  }
}