// Simplificado para AVI Lab
export interface VoiceAnalyzeRes {
  latencyIndex: number;
  pitchVar: number;
  disfluencyRate: number;
  energyStability: number;
  honestyLexicon: number;
  voiceScore: number; // 0..1
  flags: string[];
  decision: 'GO'|'REVIEW'|'NO-GO';
}

export interface VoiceEvaluateRes extends VoiceAnalyzeRes {
  transcript: string;
  words: string[];
  questionId?: string;
  contextId?: string;
}

