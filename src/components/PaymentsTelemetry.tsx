import { useMemo } from 'react';
import styled from 'styled-components';
import { Bar, BarChart, CartesianGrid, Legend, Line, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import type { DriverState } from '../types';
import { Card } from './Card';

interface PaymentsTelemetryProps {
  driverStates: DriverState[];
  filtersActive: boolean;
  totalCount: number;
}

const ChartWrapper = styled.div`
  width: 100%;
  height: 360px;
`;

const EmptyState = styled.div`
  padding: 24px;
  border-radius: ${({ theme }) => theme.radii.md};
  border: 1px dashed ${({ theme }) => theme.colors.border};
  background: rgba(15, 118, 110, 0.08);
  color: ${({ theme }) => theme.colors.textSecondary};
  text-align: center;
`;

export function PaymentsTelemetry({ driverStates, filtersActive, totalCount }: PaymentsTelemetryProps) {
  const aggregates = useMemo(() => {
    const grouped = new Map<
      string,
      { scenario: string; bankTransfer: number; cashCollection: number; coverage: number; count: number }
    >();

    driverStates.forEach((state) => {
      const key = state.scenario || 'baseline';
      const record = grouped.get(key) ?? {
        scenario: key,
        bankTransfer: 0,
        cashCollection: 0,
        coverage: 0,
        count: 0,
      };

      record.bankTransfer += Number(state.bank_transfer ?? 0);
      record.cashCollection += Number(state.cash_collection ?? 0);
      record.coverage += Number(state.coverage_ratio_30d ?? 0);
      record.count += 1;

      grouped.set(key, record);
    });

    return Array.from(grouped.values())
      .map((item) => ({
        scenario: item.scenario,
        bankTransfer: Math.round(item.bankTransfer),
        cashCollection: Math.round(item.cashCollection),
        coverage: item.count ? item.coverage / item.count : 0,
      }))
      .sort((a, b) => a.scenario.localeCompare(b.scenario));
  }, [driverStates]);

  if (filtersActive && driverStates.length === 0) {
    return (
      <Card
        title="Pagos y Telemetría"
        subtitle="Comparativo de cobranza vs cobertura promedio por escenario sintético."
      >
        <EmptyState>No hay registros que coincidan con los filtros actuales.</EmptyState>
      </Card>
    );
  }

  const subtitle = filtersActive
    ? `Comparativo filtrado (${driverStates.length} placas). Total catálogo: ${totalCount}`
    : `Comparativo de cobranza vs cobertura promedio (${totalCount} placas).`;

  return (
    <Card title="Pagos y Telemetría" subtitle={subtitle}>
      <ChartWrapper>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={aggregates} margin={{ top: 16, right: 24, bottom: 24, left: 16 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(148, 163, 184, 0.2)" />
            <XAxis dataKey="scenario" stroke="rgba(226, 232, 240, 0.6)" />
            <YAxis
              yAxisId="left"
              orientation="left"
              stroke="rgba(226, 232, 240, 0.6)"
              tickFormatter={(value) => `$${Math.round((value as number) / 1000)}k`}
            />
            <YAxis
              yAxisId="right"
              orientation="right"
              stroke="rgba(226, 232, 240, 0.6)"
              tickFormatter={(value) => `${Math.round((value as number) * 100)}%`}
              domain={[0, 1.2]}
            />
            <Tooltip
              cursor={{ strokeDasharray: '4 4' }}
              contentStyle={{
                background: '#1f2937',
                borderRadius: 12,
                border: '1px solid rgba(148,163,184,0.24)',
              }}
              formatter={(value: number, name) => {
                if (name === 'coverage') {
                  return [`${Math.round(value * 100)}%`, 'Cobertura 30d'];
                }
                return [`$${Math.round(value).toLocaleString('es-MX')}`, name === 'bankTransfer' ? 'Transferencia' : 'Cobranza'];
              }}
            />
            <Legend />
            <Bar yAxisId="left" dataKey="bankTransfer" fill="#0f766e" name="Transferencia" />
            <Bar yAxisId="left" dataKey="cashCollection" fill="#f59e0b" name="Cobranza" />
            <Line
              yAxisId="right"
              type="monotone"
              dataKey="coverage"
              stroke="#22c55e"
              strokeWidth={2}
              dot={{ r: 4, stroke: '#22c55e', strokeWidth: 1, fill: '#0b1120' }}
              name="Cobertura 30d"
            />
          </BarChart>
        </ResponsiveContainer>
      </ChartWrapper>
    </Card>
  );
}
