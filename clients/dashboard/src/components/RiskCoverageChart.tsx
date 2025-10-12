import { useMemo } from 'react';
import styled from 'styled-components';
import {
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from 'recharts';
import type { DriverState } from '../types';
import { Card } from './Card';
import { Badge } from './Badge';

interface RiskCoverageChartProps {
  driverStates: DriverState[];
  selectedPlaca?: string;
  onSelectPlaca?: (placa: string) => void;
}

const baseScenarioColor: Record<string, string> = {
  baseline: '#0f766e',
  consumption_gap: '#f59e0b',
  delinquency: '#f97316',
  fault_alert: '#ef4444',
};

const Legend = styled.ul`
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  margin: 20px 0 0;
  padding: 0;
  list-style: none;

  li {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-size: 13px;
    color: ${({ theme }) => theme.colors.textSecondary};
  }

  span {
    width: 10px;
    height: 10px;
    border-radius: 999px;
  }
`;

const EmptyState = styled.div`
  padding: 24px;
  border-radius: ${({ theme }) => theme.radii.md};
  border: 1px dashed ${({ theme }) => theme.colors.border};
  background: rgba(15, 118, 110, 0.08);
  color: ${({ theme }) => theme.colors.textSecondary};
  text-align: center;
`;

export function RiskCoverageChart({ driverStates, selectedPlaca, onSelectPlaca }: RiskCoverageChartProps) {
  const data = useMemo(
    () =>
      driverStates.map((item) => ({
        placa: item.placa,
        scenario: item.scenario,
        coverage14d: Math.max(0, Number(item.coverage_ratio_14d ?? 0)),
        riskScore: Number(item.risk_story ? riskStoryToScore(item.risk_story) : item.hase_consumption_gap_flag ?? 0),
        protectionsRemaining: Math.max(0, Number(item.protections_remaining ?? 0)),
        arrearsAmount: Math.max(0, Number(item.arrears_amount ?? 0)),
        fill: baseScenarioColor[item.scenario] ?? baseScenarioColor.baseline,
        isSelected: selectedPlaca ? item.placa === selectedPlaca : false,
      })),
    [driverStates, selectedPlaca],
  );

  const legendItems = useMemo(() => {
    const entries = new Map<string, string>();
    data.forEach((item) => {
      const key = item.scenario || 'sin-escenario';
      if (!entries.has(key)) {
        entries.set(key, baseScenarioColor[item.scenario] ?? baseScenarioColor.baseline);
      }
    });
    return Array.from(entries.entries());
  }, [data]);

  const coverageDomain = useMemo(() => {
    const max = data.reduce((acc, item) => Math.max(acc, item.coverage14d), 0);
    const padded = Math.max(1, Math.ceil(max * 10) / 10 + 0.1);
    return [0, padded] as [number, number];
  }, [data]);

  if (!data.length) {
    return (
      <Card title="Riesgo vs Cobertura" subtitle="Cobertura 14d vs severidad para cada escenario sintetizado.">
        <EmptyState>No hay registros que coincidan con los filtros actuales.</EmptyState>
      </Card>
    );
  }

  return (
    <Card
      title="Riesgo vs Cobertura"
      subtitle="Cobertura 14d vs severidad para cada escenario sintetizado."
    >
      <div style={{ width: '100%', height: 360 }}>
        <ResponsiveContainer width="100%" height={360}>
          <ScatterChart margin={{ top: 16, right: 32, bottom: 24, left: 16 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(148, 163, 184, 0.24)" />
            <XAxis
              dataKey="coverage14d"
              name="Cobertura 14d"
              type="number"
              domain={coverageDomain}
              tickFormatter={(value) => `${Math.round(value * 100)}%`}
              stroke="rgba(226, 232, 240, 0.6)"
            />
            <YAxis
              dataKey="protectionsRemaining"
              name="Protecciones restantes"
              type="number"
              allowDecimals={false}
              stroke="rgba(226, 232, 240, 0.6)"
            />
            <ZAxis dataKey="arrearsAmount" range={[60, 240]} name="Arrears" />
            <Tooltip
              cursor={{ strokeDasharray: '4 4' }}
              contentStyle={{ background: '#1f2937', borderRadius: 12, border: '1px solid rgba(148,163,184,0.24)' }}
              formatter={(value: number, name) => {
                if (name === 'coverage14d') return [`${Math.round(value * 100)}%`, 'Cobertura 14d'];
                if (name === 'protectionsRemaining') return [value, 'Protecciones'];
                if (name === 'arrearsAmount') return [`$${Math.round(value).toLocaleString('es-MX')}`, 'Arrears'];
                if (name === 'riskScore') return [value.toFixed(2), 'Score riesgo'];
                return [value, name];
              }}
              labelFormatter={(label, payload) => payload?.[0]?.payload?.placa ?? String(label)}
            />
            <Scatter
              data={data}
              shape="circle"
              name="Escenarios"
              fillOpacity={0.85}
              stroke="#0b1120"
              onClick={(point) => {
                const placa = (point as ScatterPoint)?.placa ?? (point as ScatterClickEvent)?.payload?.placa;
                if (placa && onSelectPlaca) {
                  onSelectPlaca(placa);
                }
              }}
            >
              {data.map((entry) => (
                <Cell
                  key={entry.placa}
                  fill={entry.fill}
                  fillOpacity={selectedPlaca && !entry.isSelected ? 0.2 : 0.85}
                  stroke={entry.isSelected ? '#e2e8f0' : 'rgba(15, 23, 42, 0.42)'}
                  strokeWidth={entry.isSelected ? 2.6 : 1.2}
                />
              ))}
            </Scatter>
          </ScatterChart>
        </ResponsiveContainer>
      </div>
      <Legend>
        {legendItems.map(([scenario, color]) => (
          <li key={scenario}>
            <span style={{ backgroundColor: color }} />
            <Badge tone="neutral">{scenario}</Badge>
          </li>
        ))}
      </Legend>
    </Card>
  );
}

type ScatterClickEvent = {
  payload?: {
    placa?: string;
  };
};

type ScatterPoint = {
  placa?: string;
};

function riskStoryToScore(value: string): number {
  switch (value) {
    case 'baseline':
      return 0.3;
    case 'consumption_gap':
      return 0.6;
    case 'delinquency':
      return 0.85;
    case 'fault_alert':
      return 0.7;
    default:
      return 0.5;
  }
}
