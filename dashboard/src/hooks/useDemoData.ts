import { useQuery } from '@tanstack/react-query';
import type {
  DemoDataset,
  DriverState,
  FeatureRow,
  LlmAlert,
  OutcomeLogRow,
  OutcomeScenarioSummary,
  PlanSummaryRow,
} from '../types';
import { fetchCsv, fetchJsonl } from '../utils/fetchers';
import { coerceBoolean, coerceNumber, safeJsonParse } from '../utils/parse';

type RawDriverState = Record<string, unknown>;
type RawPlanSummary = Record<string, unknown>;
type RawOutcomeLog = Record<string, unknown>;
type RawFeatureRow = Record<string, unknown>;

function normalizeDriverState(row: RawDriverState): DriverState {
  return {
    placa: String(row.placa ?? ''),
    market: String(row.market ?? ''),
    plan_type: String(row.plan_type ?? ''),
    balance: coerceNumber(row.balance),
    payment: coerceNumber(row.payment),
    term_months: coerceNumber(row.term_months),
    protections_allowed: coerceNumber(row.protections_allowed),
    protections_used: coerceNumber(row.protections_used),
    protections_remaining: coerceNumber(row.protections_remaining),
    contract_status: String(row.contract_status ?? ''),
    contract_valid_until: String(row.contract_valid_until ?? ''),
    contract_reset_cycle_days: coerceNumber(row.contract_reset_cycle_days),
    requires_manual_review: coerceBoolean(row.requires_manual_review),
    expected_payment: coerceNumber(row.expected_payment),
    bank_transfer: coerceNumber(row.bank_transfer),
    cash_collection: coerceNumber(row.cash_collection),
    observed_payment: coerceNumber(row.observed_payment),
    arrears_amount: coerceNumber(row.arrears_amount),
    coverage_ratio_30d: coerceNumber(row.coverage_ratio_30d),
    coverage_ratio_14d: coerceNumber(row.coverage_ratio_14d),
    coverage_ratio_7d: coerceNumber(row.coverage_ratio_7d),
    downtime_hours_30d: coerceNumber(row.downtime_hours_30d),
    downtime_hours_14d: coerceNumber(row.downtime_hours_14d),
    downtime_hours_7d: coerceNumber(row.downtime_hours_7d),
    distance_km_30d: coerceNumber(row.distance_km_30d),
    engine_hours_30d: coerceNumber(row.engine_hours_30d),
    fault_events_30d: coerceNumber(row.fault_events_30d),
    hase_consumption_gap_flag: coerceNumber(row.hase_consumption_gap_flag),
    hase_fault_alert_flag: coerceNumber(row.hase_fault_alert_flag),
    hase_telemetry_ok_flag: coerceNumber(row.hase_telemetry_ok_flag),
    has_recent_promise_break: coerceNumber(row.has_recent_promise_break),
    last_protection_at: String(row.last_protection_at ?? ''),
    scenario: String(row.scenario ?? ''),
    risk_story: String(row.risk_story ?? ''),
  };
}

function normalizePlanSummary(row: RawPlanSummary): PlanSummaryRow {
  return {
    plan_type: String(row.plan_type ?? ''),
    contratos: coerceNumber(row.contratos),
    protecciones_restantes_promedio: coerceNumber(row.protecciones_restantes_promedio),
    protecciones_restantes_mediana: coerceNumber(row.protecciones_restantes_mediana),
    contratos_manual: coerceNumber(row.contratos_manual),
    contratos_expirados: coerceNumber(row.contratos_expirados),
    contratos_negative: coerceNumber(row.contratos_negative),
  };
}

function normalizeOutcome(row: RawOutcomeLog): OutcomeLogRow {
  return {
    timestamp: String(row.timestamp ?? ''),
    placa: String(row.placa ?? ''),
    plaza: String(row.plaza ?? ''),
    action: String(row.action ?? ''),
    outcome: String(row.outcome ?? ''),
    reason: String(row.reason ?? ''),
    template: String(row.template ?? ''),
    risk_band: String(row.risk_band ?? ''),
    scenario: String(row.scenario ?? ''),
    notes: String(row.notes ?? ''),
    details_json: String(row.details_json ?? ''),
    metadata_json: String(row.metadata_json ?? ''),
    plan_type: String(row.plan_type ?? ''),
    protections_used: coerceNumber(row.protections_used),
    protections_allowed: coerceNumber(row.protections_allowed),
    plan_status: String(row.plan_status ?? ''),
    plan_valid_until: String(row.plan_valid_until ?? ''),
    plan_reset_cycle_days: coerceNumber(row.plan_reset_cycle_days),
    plan_requires_manual_review: coerceBoolean(row.plan_requires_manual_review),
  };
}

interface ParsedScenario {
  protection_scenario?: {
    annual_irr?: number;
    payment_change?: number;
    payment_change_pct?: number;
    term_change?: number;
    requires_manual_review?: boolean;
  };
}

function buildOutcomeScenario(row: OutcomeLogRow): OutcomeScenarioSummary {
  const parsed = safeJsonParse<ParsedScenario>(row.details_json);
  const scenario = parsed?.protection_scenario;
  return {
    timestamp: row.timestamp,
    placa: row.placa,
    action: row.action,
    scenario: row.scenario || row.template,
    annualIrr: scenario?.annual_irr ?? null,
    paymentChange: scenario?.payment_change ?? null,
    paymentChangePct: scenario?.payment_change_pct ?? null,
    termChange: scenario?.term_change ?? null,
    requiresManualReview:
      scenario?.requires_manual_review ?? coerceBoolean(row.plan_requires_manual_review),
  };
}

function normalizeFeatureRow(row: RawFeatureRow): FeatureRow {
  return {
    placa: String(row.placa ?? ''),
    outcomes_total: coerceNumber(row.outcomes_total),
    last_outcome_at: String(row.last_outcome_at ?? ''),
    days_since_last_outcome: coerceNumber(row.days_since_last_outcome),
    last_plan_type: String(row.last_plan_type ?? ''),
    last_protections_used: coerceNumber(row.last_protections_used),
    last_protections_allowed: coerceNumber(row.last_protections_allowed),
    last_plan_status: String(row.last_plan_status ?? ''),
    last_plan_valid_until: String(row.last_plan_valid_until ?? ''),
    last_plan_reset_cycle_days: coerceNumber(row.last_plan_reset_cycle_days),
    last_plan_requires_manual_review: coerceBoolean(row.last_plan_requires_manual_review),
    protections_remaining: coerceNumber(row.protections_remaining),
    protections_flag_negative: coerceBoolean(row.protections_flag_negative),
    protections_flag_expired: coerceBoolean(row.protections_flag_expired),
    protections_flag_manual: coerceBoolean(row.protections_flag_manual),
  };
}

function normalizeAlert(alert: LlmAlert): LlmAlert {
  const content = alert.content?.replace(/\\n/g, '\n') ?? '';
  const timestamp = alert.timestamp ?? alert?.context?.reference_ts ?? '';
  return {
    ...alert,
    timestamp,
    content,
  };
}

const DATA_BASE = (import.meta.env.VITE_DEMO_DATA_BASE ?? '/data').replace(/\/$/, '');
const REPORTS_BASE = (import.meta.env.VITE_DEMO_REPORTS_BASE ?? DATA_BASE).replace(/\/$/, '');

export function useDemoData() {
  return useQuery<DemoDataset, Error>({
    queryKey: ['demo-dataset'],
    queryFn: async () => {
      const [driverStatesRaw, planSummaryRaw, outcomeLogsRaw, featuresRaw, alertsRaw] = await Promise.all([
        fetchCsv<RawDriverState>(`${DATA_BASE}/synthetic_driver_states.csv`),
        fetchCsv<RawPlanSummary>(`${DATA_BASE}/pia_plan_summary.csv`),
        fetchCsv<RawOutcomeLog>(`${DATA_BASE}/pia_outcomes_log.csv`),
        fetchCsv<RawFeatureRow>(`${DATA_BASE}/pia_outcomes_features.csv`),
        fetchJsonl<LlmAlert>(`${REPORTS_BASE}/pia_llm_outbox.jsonl`, { optional: true }),
      ]);

      const outcomeLogs = outcomeLogsRaw.map(normalizeOutcome);
      const outcomeScenarios = outcomeLogs.map(buildOutcomeScenario).filter((item) => item.annualIrr !== null);

      const dataset: DemoDataset = {
        driverStates: driverStatesRaw.map(normalizeDriverState),
        planSummary: planSummaryRaw.map(normalizePlanSummary),
        outcomeLogs,
        outcomeScenarios,
        features: featuresRaw.map(normalizeFeatureRow),
        alerts: alertsRaw.map(normalizeAlert),
      };

      return dataset;
    },
    staleTime: 10 * 60 * 1000,
  });
}
