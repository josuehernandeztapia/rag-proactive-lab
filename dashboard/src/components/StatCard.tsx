import styled from 'styled-components';
import { Card } from './Card';

type StatProps = {
  label: string;
  value: string;
  trendLabel?: string;
  trend?: 'up' | 'down' | 'flat';
  helper?: string;
};

const Label = styled.p`
  font-size: 14px;
  color: ${({ theme }) => theme.colors.textSecondary};
  margin: 0 0 8px;
`;

const Metric = styled.div`
  font-size: 32px;
  font-weight: 700;
`;

const Trend = styled.span<{ trend: 'up' | 'down' | 'flat' }>`
  display: inline-flex;
  align-items: center;
  margin-top: 6px;
  font-size: 14px;
  color: ${({ trend, theme }) => {
    switch (trend) {
      case 'up':
        return theme.colors.success;
      case 'down':
        return theme.colors.danger;
      default:
        return theme.colors.textSecondary;
    }
  }};
`;

const Helper = styled.span`
  display: block;
  margin-top: 6px;
  font-size: 12px;
  color: ${({ theme }) => theme.colors.textSecondary};
`;

export function StatCard({ label, value, trendLabel, trend = 'flat', helper }: StatProps) {
  return (
    <Card>
      <Label>{label}</Label>
      <Metric>{value}</Metric>
      {trendLabel ? <Trend trend={trend}>{trendLabel}</Trend> : null}
      {helper ? <Helper>{helper}</Helper> : null}
    </Card>
  );
}
