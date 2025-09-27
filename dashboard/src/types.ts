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
  contract_reset_cycle_days: number;
  requires_manual_review: string | boolean;
  expected_payment: number;
  bank_transfer: number;
  cash_collection: number;
  observed_payment: number;
  arrears_amount: number;
  coverage_ratio_30d: number;
  coverage_ratio_14d: number;
  coverage_ratio_7d: number;
  downtime_hours_30d: number;
  downtime_hours_14d: number;
  downtime_hours_7d: number;
  distance_km_30d: number;
  engine_hours_30d: number;
  fault_events_30d: number;
  hase_consumption_gap_flag: number;
  hase_fault_alert_flag: number;
  hase_telemetry_ok_flag: number;
  has_recent_promise_break: number;
  last_protection_at: string;
  scenario: string;
  risk_story: string;
}

export interface PlanSummaryRow {
  plan_type: string;
  contratos: number;
  protecciones_restantes_promedio: number;
  protecciones_restantes_mediana: number;
  contratos_manual: number;
  contratos_expirados: number;
  contratos_negative: number;
}

export interface OutcomeLogRow {
  timestamp: string;
  placa: string;
  plaza: string;
  action: string;
  outcome: string;
  reason: string;
  template: string;
  risk_band: string;
  scenario: string;
  notes: string;
  details_json: string;
  metadata_json: string;
  plan_type: string;
  protections_used: number;
  protections_allowed: number;
  plan_status: string;
  plan_valid_until: string;
  plan_reset_cycle_days: number;
  plan_requires_manual_review: string | boolean;
}

export interface OutcomeScenarioSummary {
  timestamp: string;
  placa: string;
  action: string;
  scenario: string;
  annualIrr: number | null;
  paymentChange: number | null;
  paymentChangePct: number | null;
  termChange: number | null;
  requiresManualReview: boolean;
}

export interface LlmAlert {
  timestamp: string;
  placa: string;
  contact: string | null;
  flags: {
    protecciones_negativas?: boolean;
    plan_expirado?: boolean;
    requiere_revision_manual?: boolean;
    [key: string]: unknown;
  };
  metric_snapshot: string;
  content: string;
  context?: {
    prompt_version?: string;
    reference_ts?: string;
    plan_type?: string;
    plan_status?: string;
    protections_remaining?: number;
    [key: string]: unknown;
  };
}

export interface FeatureRow {
  placa: string;
  outcomes_total: number;
  last_outcome_at: string;
  days_since_last_outcome: number;
  last_plan_type: string;
  last_protections_used: number;
  last_protections_allowed: number;
  last_plan_status: string;
  last_plan_valid_until: string;
  last_plan_reset_cycle_days: number;
  last_plan_requires_manual_review: string | boolean;
  protections_remaining: number;
  protections_flag_negative: string | boolean;
  protections_flag_expired: string | boolean;
  protections_flag_manual: string | boolean;
}

export interface DemoDataset {
  driverStates: DriverState[];
  planSummary: PlanSummaryRow[];
  outcomeLogs: OutcomeLogRow[];
  outcomeScenarios: OutcomeScenarioSummary[];
  features: FeatureRow[];
  alerts: LlmAlert[];
}
