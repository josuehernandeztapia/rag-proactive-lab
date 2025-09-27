import dayjs from 'dayjs';
import relativeTime from 'dayjs/plugin/relativeTime';
import 'dayjs/locale/es';
import styled from 'styled-components';
import type { DemoDataset } from '../types';
import { Card } from './Card';

dayjs.extend(relativeTime);
dayjs.locale('es');

interface RunbookPanelProps {
  data: DemoDataset;
}

const DATA_BASE = (import.meta.env.VITE_DEMO_DATA_BASE ?? '/data').replace(/\/$/, '');
const REPORTS_BASE = (import.meta.env.VITE_DEMO_REPORTS_BASE ?? DATA_BASE).replace(/\/$/, '');

const Grid = styled.div`
  display: grid;
  gap: 24px;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
`;

const Section = styled.div`
  display: flex;
  flex-direction: column;
  gap: 12px;

  h3 {
    margin: 0;
    font-size: 16px;
  }

  ul,
  ol {
    margin: 0;
    padding-left: 18px;
    display: grid;
    gap: 8px;
    color: ${({ theme }) => theme.colors.textSecondary};
    font-size: 14px;
  }

  code {
    background: rgba(15, 118, 110, 0.16);
    padding: 2px 6px;
    border-radius: ${({ theme }) => theme.radii.sm};
    font-size: 13px;
  }
`;

const MetaItem = styled.li`
  display: flex;
  flex-direction: column;
  gap: 4px;

  strong {
    color: ${({ theme }) => theme.colors.textPrimary};
  }

  span {
    color: ${({ theme }) => theme.colors.textSecondary};
    font-size: 13px;
  }
`;

export function RunbookPanel({ data }: RunbookPanelProps) {
  const lastOutcome = maxTimestamp(data.outcomeLogs.map((item) => item.timestamp));
  const lastAlert = maxTimestamp(data.alerts.map((item) => item.timestamp));

  return (
    <Card title="Runbook" subtitle="Checklist rápido para narrar la corrida sintética más reciente.">
      <Grid>
        <Section>
          <h3>Últimos artefactos</h3>
          <ul>
            <MetaItem>
              <strong>Outcomes PIA</strong>
              <span>{lastOutcome ? formatDate(lastOutcome) : 'Sin registro'}</span>
            </MetaItem>
            <MetaItem>
              <strong>Alertas LLM</strong>
              <span>{lastAlert ? formatDate(lastAlert) : 'Sin registro'}</span>
            </MetaItem>
            <MetaItem>
              <strong>Feature store</strong>
              <span>{data.features.length.toLocaleString('es-MX')} placas</span>
            </MetaItem>
          </ul>
        </Section>
        <Section>
          <h3>Archivos fuente</h3>
          <ol>
            <li><code>{`${DATA_BASE}/synthetic_driver_states.csv`}</code></li>
            <li><code>{`${DATA_BASE}/pia_outcomes_log.csv`}</code></li>
            <li><code>{`${DATA_BASE}/pia_outcomes_features.csv`}</code></li>
            <li><code>{`${DATA_BASE}/pia_plan_summary.csv`}</code></li>
            <li><code>{`${REPORTS_BASE}/pia_llm_outbox.jsonl`}</code></li>
          </ol>
        </Section>
        <Section>
          <h3>Tips demo</h3>
          <ul>
            <li>Elige tres placas (baseline, consumption_gap, delinquency) y prepara narrativa.</li>
            <li>Resalta cobertura vs protecciones en la vista de riesgo.</li>
            <li>Termina mostrando la alerta LLM tal como llegaría al asesor.</li>
          </ul>
        </Section>
      </Grid>
    </Card>
  );
}

function maxTimestamp(values: string[]): string | null {
  if (!values.length) return null;
  const valid = values.filter((value) => dayjs(value).isValid());
  if (!valid.length) return null;
  return valid.reduce((max, current) => (dayjs(current).isAfter(dayjs(max)) ? current : max));
}

function formatDate(value: string): string {
  const date = dayjs(value);
  if (!date.isValid()) return 'Sin registro';
  return `${date.format('YYYY-MM-DD HH:mm')} · ${date.fromNow()}`;
}
