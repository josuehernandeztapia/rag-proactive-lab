import dayjs from 'dayjs';
import styled from 'styled-components';
import type { LlmAlert } from '../types';
import { Card } from './Card';
import { Badge } from './Badge';

interface AlertsListProps {
  alerts: LlmAlert[];
}

const AlertsGrid = styled.div`
  display: grid;
  gap: 16px;
`;

const AlertCard = styled.article`
  border: 1px solid ${({ theme }) => theme.colors.border};
  border-radius: ${({ theme }) => theme.radii.md};
  background: ${({ theme }) => theme.colors.surface};
  padding: 18px 20px;
  display: flex;
  flex-direction: column;
  gap: 12px;
`;

const AlertHeader = styled.header`
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  gap: 12px;

  h3 {
    margin: 0;
    font-size: 16px;
  }

  time {
    font-size: 12px;
    color: ${({ theme }) => theme.colors.textSecondary};
  }
`;

const AlertContent = styled.pre`
  margin: 0;
  white-space: pre-wrap;
  font-family: 'JetBrains Mono', 'SFMono-Regular', Consolas, 'Liberation Mono', monospace;
  background: rgba(30, 41, 59, 0.8);
  border-radius: ${({ theme }) => theme.radii.sm};
  border: 1px solid rgba(148, 163, 184, 0.16);
  padding: 12px 14px;
  color: ${({ theme }) => theme.colors.textPrimary};
`;

const FlagList = styled.ul`
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin: 0;
  padding: 0;
  list-style: none;
`;

export function AlertsList({ alerts }: AlertsListProps) {
  const ordered = [...alerts].sort((a, b) => dayjs(b.timestamp).valueOf() - dayjs(a.timestamp).valueOf());

  return (
    <Card title="Alertas LLM" subtitle="Narrativas listas para el asesor según la última corrida del notifier.">
      <AlertsGrid>
        {ordered.map((alert) => (
          <AlertCard key={`${alert.timestamp}-${alert.placa ?? 'sin-placa'}`}>
            <AlertHeader>
              <h3>{alert.placa ?? 'Sin placa'}</h3>
              <time dateTime={alert.timestamp}>{dayjs(alert.timestamp).format('YYYY-MM-DD HH:mm')}</time>
            </AlertHeader>
            <AlertContent>{alert.content}</AlertContent>
            {alert.flags ? (
              <FlagList>
                {Object.entries(alert.flags)
                  .filter(([, value]) => Boolean(value))
                  .map(([flag]) => (
                    <li key={flag}>
                      <Badge tone="warning">{flag}</Badge>
                    </li>
                  ))}
              </FlagList>
            ) : null}
          </AlertCard>
        ))}
      </AlertsGrid>
    </Card>
  );
}
