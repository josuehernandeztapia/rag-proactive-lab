// AVI Types - Interfaces for the voice analysis system

export enum AVICategory {
  BASIC_INFO = 'basic_info',
  DAILY_OPERATION = 'daily_operation',
  OPERATIONAL_COSTS = 'operational_costs',
  BUSINESS_STRUCTURE = 'business_structure',
  ASSETS_PATRIMONY = 'assets_patrimony',
  CREDIT_HISTORY = 'credit_history',
  PAYMENT_INTENTION = 'payment_intention',
  RISK_EVALUATION = 'risk_evaluation'
}

export type RiskImpact = 'LOW' | 'MEDIUM' | 'HIGH';

export interface AVIAnalytics {
  expectedResponseTime: number;
  stressIndicators: string[];
  truthVerificationKeywords: string[];
}

export interface AVIQuestionEnhanced {
  id: string;
  category: AVICategory;
  question: string;
  weight: number;
  riskImpact: RiskImpact;
  stressLevel: number;
  estimatedTime: number;
  verificationTriggers: string[];
  followUpQuestions?: string[];
  analytics: AVIAnalytics;
}

export interface AVIConfig {
  total_questions: number;
  implemented_questions: number;
  remaining_questions: number;
  estimated_duration_minutes: number;
  critical_questions: number;
  high_stress_questions: number;
  completion_percentage: number;
  questions_by_category: Record<string, number>;
  questions_by_weight: Record<string, number>;
  questions_by_stress_level: Record<string, number>;
  system_status: string;
}