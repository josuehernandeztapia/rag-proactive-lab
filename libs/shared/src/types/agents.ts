// Types specific to RAG Proactive Lab agents

export interface DriverState {
  placa: string;
  market: string;
  plan_type: string;
  balance: number;
  payment: number;
  term_months: number;
  protections_allowed: number;
  protections_used: number;
  protections_remaining: number;
  contract_status: string;
  contract_valid_until: string;
  scenario?: string;
  risk_story?: string;
}

export interface GuardianAlert {
  id: string;
  generatedAt: string;
  eventTs: string | null;
  placa: string;
  alertType: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  message: string;
  summary: string;
  recommendation: string;
  contact: string | null;
  source?: string;
  insight?: Record<string, unknown> | null;
  scenario?: string;
  market?: string;
}

export interface LlmAlert {
  timestamp: string;
  content: string;
  context?: {
    reference_ts?: string;
    [key: string]: any;
  };
}

export interface AgentCapabilities {
  name: string;
  description: string;
  version: string;
  endpoints: string[];
  status: 'active' | 'inactive' | 'maintenance';
}