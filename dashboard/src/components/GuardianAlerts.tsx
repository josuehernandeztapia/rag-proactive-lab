import dayjs from 'dayjs';
import styled from 'styled-components';
import type { GuardianAlert } from '../types';
import { Card } from './Card';
import { Badge } from './Badge';

interface GuardianAlertsProps {
  alerts: GuardianAlert[];
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

const HeaderRow = styled.header`
  display: flex;
  flex-direction: column;
  gap: 8px;

  @media (min-width: 720px) {
    flex-direction: row;
    justify-content: space-between;
    align-items: baseline;
  }

  h3 {
    margin: 0;
    font-size: 16px;
    display: flex;
    align-items: center;
    gap: 8px;
  }

  time {
    font-size: 12px;
    color: ${({ theme }) => theme.colors.textSecondary};
  }
`;

const Message = styled.pre`
  margin: 0;
  white-space: pre-wrap;
  font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont;
  line-height: 1.5;
  background: rgba(15, 23, 42, 0.6);
  border-radius: ${({ theme }) => theme.radii.sm};
  border: 1px solid rgba(148, 163, 184, 0.16);
  padding: 12px 14px;
`;

const MetaRow = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  font-size: 12px;
  color: ${({ theme }) => theme.colors.textSecondary};
`;

const Summary = styled.p`
  margin: 0;
  font-size: 13px;
  color: ${({ theme }) => theme.colors.textSecondary};
`;

function mapSeverityToTone(severity: string | undefined) {
  if (!severity) return 'neutral';
  const normalized = severity.toLowerCase();
  if (normalized === 'high') return 'danger';
  if (normalized === 'medium') return 'warning';
  if (normalized === 'low') return 'success';
  return 'neutral';
}

export function GuardianAlerts({ alerts }: GuardianAlertsProps) {
  if (!alerts.length) {
    return (
      <Card title="Guardian de Flota" subtitle="Sin alertas recientes. Ejecuta make demo-proteccion o el notifier para poblar mensajes.">
        <EmptyState>No se encontraron alertas con el filtro activo.</EmptyState>
      </Card>
    );
  }

  const ordered = [...alerts].sort((a, b) => dayjs(b.generatedAt).valueOf() - dayjs(a.generatedAt).valueOf());

  return (
    <Card
      title="Guardian de Flota"
      subtitle="Alertas proactivas (downtime, consumo, DTC) listas para WhatsApp."
    >
      <AlertsGrid>
        {ordered.map((alert) => (
          <AlertCard key={alert.id}>
            <HeaderRow>
              <h3>
                {alert.placa}
                <Badge tone={mapSeverityToTone(alert.severity)}>{alert.severity?.toUpperCase() ?? 'INFO'}</Badge>
              </h3>
              <time dateTime={alert.generatedAt}>{dayjs(alert.generatedAt).format('YYYY-MM-DD HH:mm')}</time>
            </HeaderRow>
            <Summary>{alert.summary}</Summary>
            <Message>{alert.message}</Message>
            <MetaRow>
              {alert.alertType ? <span>Tipo: {alert.alertType}</span> : null}
              {alert.eventTs ? <span>Evento: {dayjs(alert.eventTs).format('YYYY-MM-DD HH:mm')}</span> : null}
              {alert.contact ? <span>Contacto: {alert.contact}</span> : null}
              {alert.scenario ? <span>Escenario: {alert.scenario}</span> : null}
              {alert.market ? <span>Plaza: {alert.market}</span> : null}
            </MetaRow>
          </AlertCard>
        ))}
      </AlertsGrid>
    </Card>
  );
}

const EmptyState = styled.div`
  padding: 32px 20px;
  border-radius: ${({ theme }) => theme.radii.md};
  border: 1px dashed rgba(148, 163, 184, 0.24);
  background: rgba(15, 23, 42, 0.4);
  text-align: center;
  color: ${({ theme }) => theme.colors.textSecondary};
`;
