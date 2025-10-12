import { useMemo } from 'react';
import styled from 'styled-components';
import type { DemoDataset, DriverState, OutcomeScenarioSummary } from '../types';
import { Card } from './Card';
import { StatCard } from './StatCard';

const SectionTitle = styled.div`
  display: flex;
  flex-direction: column;
  gap: 8px;

  h2 {
    margin: 0;
    font-size: 20px;
  }

  p {
    margin: 0;
    font-size: 14px;
    color: ${({ theme }) => theme.colors.textSecondary};
  }
`;

const StatGrid = styled.div`
  margin-top: 24px;
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 24px;
`;

interface PortfolioOverviewProps {
  data: DemoDataset;
  filteredDriverStates: DriverState[];
  filteredOutcomeScenarios: OutcomeScenarioSummary[];
  filtersActive: boolean;
}

export function PortfolioOverview({ data, filteredDriverStates, filteredOutcomeScenarios, filtersActive }: PortfolioOverviewProps) {
  const totalMetrics = useMemo(
    () => calculateMetrics(data.driverStates, data.outcomeScenarios),
    [data.driverStates, data.outcomeScenarios],
  );

  const filteredMetrics = useMemo(
    () => calculateMetrics(filteredDriverStates, filteredOutcomeScenarios),
    [filteredDriverStates, filteredOutcomeScenarios],
  );

  const display = filtersActive ? filteredMetrics : totalMetrics;

  return (
    <Card>
      <SectionTitle>
        <h2>Portfolio Overview</h2>
        <p>Contratos activos, protecciones y TIR promedio listos para narrar el showcase.</p>
      </SectionTitle>
      <StatGrid>
        <StatCard
          label="Contratos activos"
          value={display.activeContracts.toLocaleString('es-MX')}
          helper={filtersActive ? `Total: ${totalMetrics.activeContracts.toLocaleString('es-MX')}` : undefined}
        />
        <StatCard
          label="Protecciones promedio"
          value={display.protectionsAvg.toFixed(2)}
          helper={filtersActive ? `Total: ${totalMetrics.protectionsAvg.toFixed(2)}` : undefined}
        />
        <StatCard
          label="Revisión manual"
          value={display.manualReview.toLocaleString('es-MX')}
          trend={display.manualReview > 0 ? 'down' : 'flat'}
          trendLabel={display.manualReview > 0 ? 'Atiende banderas' : 'Sin banderas críticas'}
          helper={filtersActive ? `Total: ${totalMetrics.manualReview.toLocaleString('es-MX')}` : undefined}
        />
        <StatCard
          label="Cobertura 14d promedio"
          value={`${(display.coverageAvg14d * 100).toFixed(1)} %`}
          helper={filtersActive ? `Total: ${(totalMetrics.coverageAvg14d * 100).toFixed(1)} %` : undefined}
        />
        <StatCard
          label="Arrears totales"
          value={`$${Math.round(display.arrearsTotal).toLocaleString('es-MX')}`}
          helper={filtersActive ? `$${Math.round(totalMetrics.arrearsTotal).toLocaleString('es-MX')}` : undefined}
        />
        <StatCard
          label="Planes expirados"
          value={display.expiredContracts.toLocaleString('es-MX')}
          trend={display.expiredContracts > 0 ? 'down' : 'flat'}
          trendLabel={display.expiredContracts > 0 ? 'Revisión urgente' : 'Vigencia al día'}
          helper={filtersActive ? `Total: ${totalMetrics.expiredContracts.toLocaleString('es-MX')}` : undefined}
        />
        <StatCard
          label="TIR media"
          value={`${(display.avgIrr * 100).toFixed(2)} %`}
          helper={filtersActive ? `Total: ${(totalMetrics.avgIrr * 100).toFixed(2)} %` : undefined}
        />
      </StatGrid>
    </Card>
  );
}

function calculateMetrics(driverStates: DriverState[], outcomeScenarios: OutcomeScenarioSummary[]) {
  const activeContracts = driverStates.filter((item) => item.contract_status === 'active').length;
  const protectionsAvg = driverStates.length
    ? driverStates.reduce((acc, item) => acc + item.protections_remaining, 0) / driverStates.length
    : 0;
  const manualReview = driverStates.filter((item) => Boolean(item.requires_manual_review)).length;
  const expiredContracts = driverStates.filter((item) => item.contract_status !== 'active').length;
  const irrValues = outcomeScenarios
    .map((item) => item.annualIrr)
    .filter((value): value is number => value !== null && value !== undefined);
  const coverageAvg14d = driverStates.length
    ? driverStates.reduce((acc, item) => acc + (item.coverage_ratio_14d || 0), 0) / driverStates.length
    : 0;
  const arrearsTotal = driverStates.reduce((acc, item) => acc + Math.max(item.arrears_amount || 0, 0), 0);
  const avgIrr = irrValues.length ? irrValues.reduce((acc, value) => acc + value, 0) / irrValues.length : 0;

  return {
    activeContracts,
    protectionsAvg,
    manualReview,
    expiredContracts,
    coverageAvg14d,
    arrearsTotal,
    avgIrr,
  };
}
