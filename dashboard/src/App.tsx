import { useEffect, useMemo, useState } from 'react';
import dayjs from 'dayjs';
import styled from 'styled-components';
import { AlertsList } from './components/AlertsList';
import { DashboardFilters } from './components/DashboardFilters';
import { DashboardLayout } from './components/DashboardLayout';
import { PaymentsTelemetry } from './components/PaymentsTelemetry';
import { PortfolioOverview } from './components/PortfolioOverview';
import { ProtectionHeatmap } from './components/ProtectionHeatmap';
import { RiskCoverageChart } from './components/RiskCoverageChart';
import { RunbookPanel } from './components/RunbookPanel';
import { OutcomeTable } from './components/OutcomeTable';
import { useDemoData } from './hooks/useDemoData';
import type { OutcomeScenarioSummary } from './types';

dayjs.locale('es');

type DateRange = { start: string; end: string };

function App() {
  const { data, isLoading, error, refetch, isFetching } = useDemoData();
  const [scenarioFilter, setScenarioFilter] = useState('all');
  const [plateFilter, setPlateFilter] = useState('all');
  const [plazaFilter, setPlazaFilter] = useState('all');
  const [dateRange, setDateRange] = useState<DateRange>({ start: '', end: '' });

  const driverStates = data?.driverStates ?? [];
  const outcomeScenarios = data?.outcomeScenarios ?? [];

  const allowedPlacasByDate = useMemo(
    () => buildDateFilterSet(outcomeScenarios, dateRange.start, dateRange.end),
    [outcomeScenarios, dateRange.start, dateRange.end],
  );
  const hasDateFilter = Boolean(dateRange.start || dateRange.end);

  const scenarioOptions = useMemo(() => {
    const unique = new Map<string, string>();
    driverStates.forEach((item) => {
      const key = item.scenario || 'sin-escenario';
      if (!unique.has(key)) unique.set(key, formatScenario(key));
    });
    return [
      { value: 'all', label: 'Todos los escenarios' },
      ...Array.from(unique.entries()).map(([value, label]) => ({ value, label })),
    ];
  }, [driverStates]);

  const plazaOptions = useMemo(() => {
    const unique = new Map<string, string>();
    driverStates.forEach((item) => {
      unique.set(item.market || 'sin-plaza', item.market || 'Sin plaza');
    });
    return [
      { value: 'all', label: 'Todas las plazas' },
      ...Array.from(unique.entries()).map(([value, label]) => ({ value, label })),
    ];
  }, [driverStates]);

  const plateOptions = useMemo(() => {
    const scopedDriverStates = driverStates.filter((item) => {
      const matchesScenario =
        scenarioFilter === 'all' || (item.scenario || 'sin-escenario') === scenarioFilter;
      const matchesPlaza = plazaFilter === 'all' || item.market === plazaFilter;
      const matchesDate = !hasDateFilter || allowedPlacasByDate.has(item.placa);
      return matchesScenario && matchesPlaza && matchesDate;
    });
    const unique = new Map<string, string>();
    scopedDriverStates.forEach((item) => {
      unique.set(item.placa, item.placa);
    });
    return [
      { value: 'all', label: 'Todas las placas' },
      ...Array.from(unique.entries()).map(([value, label]) => ({ value, label })),
    ];
  }, [driverStates, scenarioFilter, plazaFilter, allowedPlacasByDate, hasDateFilter]);

  useEffect(() => {
    if (!plateOptions.some((option) => option.value === plateFilter)) {
      setPlateFilter('all');
    }
    if (!plazaOptions.some((option) => option.value === plazaFilter)) {
      setPlazaFilter('all');
    }
  }, [plateOptions, plazaOptions, plateFilter, plazaFilter]);

  const driverMap = useMemo(() => {
    return new Map(driverStates.map((item) => [item.placa, item]));
  }, [driverStates]);

  const filteredOutcomeScenarios = useMemo(() => {
    return outcomeScenarios.filter((item) => {
      const matchesScenario =
        scenarioFilter === 'all' || (item.scenario || 'sin-escenario') === scenarioFilter;
      const matchesPlate = plateFilter === 'all' || item.placa === plateFilter;
      const matchesPlaza = plazaFilter === 'all' || driverMap.get(item.placa)?.market === plazaFilter;
      const matchesDate = isWithinRange(item.timestamp, dateRange.start, dateRange.end);
      return matchesScenario && matchesPlate && matchesPlaza && matchesDate;
    });
  }, [outcomeScenarios, scenarioFilter, plateFilter, plazaFilter, driverMap, dateRange.start, dateRange.end]);

  const filteredDriverStates = useMemo(() => {
    return driverStates.filter((item) => {
      const matchesScenario =
        scenarioFilter === 'all' || (item.scenario || 'sin-escenario') === scenarioFilter;
      const matchesPlate = plateFilter === 'all' || item.placa === plateFilter;
      const matchesPlaza = plazaFilter === 'all' || item.market === plazaFilter;
      const matchesDate = !hasDateFilter || allowedPlacasByDate.has(item.placa);
      return matchesScenario && matchesPlate && matchesPlaza && matchesDate;
    });
  }, [driverStates, scenarioFilter, plateFilter, plazaFilter, allowedPlacasByDate, hasDateFilter]);

  const handleSelectPlate = (placa: string) => {
    setPlateFilter((prev) => (prev === placa ? 'all' : placa));
  };

  const handleClearFilters = () => {
    setScenarioFilter('all');
    setPlateFilter('all');
    setPlazaFilter('all');
    setDateRange({ start: '', end: '' });
  };

  const filtersActive =
    scenarioFilter !== 'all' || plateFilter !== 'all' || plazaFilter !== 'all' || hasDateFilter;
  const selectedPlate = plateFilter === 'all' ? undefined : plateFilter;

  const sidebarLinks = [
    { href: '#portfolio-overview', label: 'Overview' },
    { href: '#risk-vs-coverage', label: 'Riesgo' },
    { href: '#protection-heatmap', label: 'Protecciones' },
    { href: '#tir-drilldown', label: 'TIR' },
    { href: '#payments-telemetry', label: 'Pagos' },
    { href: '#alerts-llm', label: 'Alertas' },
    { href: '#runbook', label: 'Runbook' },
  ];

  return (
    <DashboardLayout sidebarLinks={sidebarLinks}>
      <HeaderSection>
        <Title>Dashboard Demo Protección</Title>
        <Description>
          Visualización rápida de la corrida sintética (<code>make demo-proteccion</code>) con focos en riesgo,
          cobertura, TIR y storytelling LLM.
        </Description>
        <ActionsRow>
          <RefreshButton type="button" onClick={() => refetch()} disabled={isFetching}>
            {isFetching ? 'Actualizando…' : 'Refrescar datos'}
          </RefreshButton>
          <AnchorNav>
            {sidebarLinks.map((link) => (
              <a key={link.href} href={link.href}>
                {link.label}
              </a>
            ))}
          </AnchorNav>
        </ActionsRow>
        {data ? (
          <FiltersSection>
            <DashboardFilters
              scenarioOptions={scenarioOptions}
              plateOptions={plateOptions}
              plazaOptions={plazaOptions}
              scenarioValue={scenarioFilter}
              plateValue={plateFilter}
              plazaValue={plazaFilter}
              startDate={dateRange.start}
              endDate={dateRange.end}
              onScenarioChange={setScenarioFilter}
              onPlateChange={setPlateFilter}
              onPlazaChange={setPlazaFilter}
              onDateRangeChange={(field, value) =>
                setDateRange((prev) => ({ ...prev, [field]: value }))
              }
              onClear={handleClearFilters}
            />
            <FilterSummary>
              <SummaryItem>
                <label>Escenario</label>
                <span>{formatScenarioLabel(scenarioFilter)}</span>
              </SummaryItem>
              <SummaryItem>
                <label>Placa</label>
                <span>{plateFilter === 'all' ? 'Todas' : plateFilter}</span>
              </SummaryItem>
              <SummaryItem>
                <label>Plaza</label>
                <span>{plazaFilter === 'all' ? 'Todas' : plazaFilter}</span>
              </SummaryItem>
              <SummaryItem>
                <label>Rango</label>
                <span>
                  {dateRange.start ? dayjs(dateRange.start).format('YYYY-MM-DD') : '—'} →{' '}
                  {dateRange.end ? dayjs(dateRange.end).format('YYYY-MM-DD') : '—'}
                </span>
              </SummaryItem>
              {filtersActive ? (
                <ClearFiltersButton type="button" onClick={handleClearFilters}>
                  Limpiar
                </ClearFiltersButton>
              ) : null}
            </FilterSummary>
          </FiltersSection>
        ) : null}
      </HeaderSection>
      <MainStack>
        {isLoading ? (
          <StatusMessage>Cargando datasets…</StatusMessage>
        ) : error ? (
          <StatusMessage tone="danger">{error.message}</StatusMessage>
        ) : data ? (
          <>
            <section id="portfolio-overview">
              <PortfolioOverview
                data={data}
                filteredDriverStates={filteredDriverStates}
                filteredOutcomeScenarios={filteredOutcomeScenarios}
                filtersActive={filtersActive}
              />
            </section>
            <section id="risk-vs-coverage">
              <RiskCoverageChart
                driverStates={filteredDriverStates}
                selectedPlaca={selectedPlate}
                onSelectPlaca={handleSelectPlate}
              />
            </section>
            <section id="protection-heatmap">
              <ProtectionHeatmap planSummary={data.planSummary} />
            </section>
            <section id="tir-drilldown">
              <OutcomeTable
                data={filteredOutcomeScenarios}
                selectedPlaca={selectedPlate}
                onSelectPlaca={handleSelectPlate}
              />
            </section>
            <section id="payments-telemetry">
              <PaymentsTelemetry
                driverStates={filtersActive ? filteredDriverStates : data.driverStates}
                filtersActive={filtersActive}
                totalCount={data.driverStates.length}
              />
            </section>
            <section id="alerts-llm">
              <AlertsList alerts={data.alerts} />
            </section>
            <section id="runbook">
              <RunbookPanel data={data} />
            </section>
          </>
        ) : null}
      </MainStack>
      <FooterNote>
        Datos: <code>make demo-proteccion</code> · Guías: docs/demo_runbook_hase_pia_tir_proteccion.md y
        docs/hus_dashboard_proteccion.md
      </FooterNote>
    </DashboardLayout>
  );
}

function buildDateFilterSet(outcomes: OutcomeScenarioSummary[], start: string, end: string): Set<string> {
  if (!outcomes.length) return new Set();
  const set = new Set<string>();
  outcomes.forEach((item) => {
    if (isWithinRange(item.timestamp, start, end)) {
      set.add(item.placa);
    }
  });
  if (!start && !end) {
    // si no hay filtro, agregar todas las placas conocidas
    outcomes.forEach((item) => set.add(item.placa));
  }
  return set;
}

function isWithinRange(timestamp: string, start: string, end: string): boolean {
  if (!start && !end) return true;
  const ts = dayjs(timestamp);
  if (!ts.isValid()) return false;
  if (start && ts.isBefore(dayjs(start), 'day')) return false;
  if (end && ts.isAfter(dayjs(end), 'day')) return false;
  return true;
}

function formatScenario(raw: string): string {
  if (!raw || raw === 'sin-escenario') return 'Sin escenario';
  const normalized = raw.replace(/_/g, ' ').toLowerCase();
  return normalized.replace(/(^|\s)\w/g, (match) => match.toUpperCase());
}

function formatScenarioLabel(value: string): string {
  if (value === 'all') return 'Todos';
  return formatScenario(value);
}

export default App;

const HeaderSection = styled.header`
  display: flex;
  flex-direction: column;
  gap: 18px;
`;

const Title = styled.h1`
  margin: 0;
  font-size: 32px;
`;

const Description = styled.p`
  margin: 0;
  color: ${({ theme }) => theme.colors.textSecondary};
  font-size: 15px;

  code {
    background: rgba(15, 118, 110, 0.16);
    padding: 2px 6px;
    border-radius: ${({ theme }) => theme.radii.sm};
  }
`;

const ActionsRow = styled.div`
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 16px;
`;

const RefreshButton = styled.button`
  border-radius: ${({ theme }) => theme.radii.sm};
  border: 1px solid transparent;
  background: ${({ theme }) => theme.colors.primary};
  color: #0f172a;
  padding: 10px 18px;
  font-size: 14px;
  font-weight: 600;
  cursor: pointer;
  transition: transform 0.2s ease, box-shadow 0.2s ease;

  &:hover:not(:disabled) {
    transform: translateY(-1px);
    box-shadow: 0 12px 32px -16px rgba(15, 118, 110, 0.8);
  }

  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
`;

const AnchorNav = styled.nav`
  display: flex;
  flex-wrap: wrap;
  gap: 12px;

  a {
    padding: 6px 12px;
    border-radius: 999px;
    border: 1px solid rgba(148, 163, 184, 0.24);
    color: ${({ theme }) => theme.colors.textSecondary};
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;

    &:hover {
      color: ${({ theme }) => theme.colors.textPrimary};
      border-color: ${({ theme }) => theme.colors.primary};
    }
  }
`;

const MainStack = styled.main`
  display: flex;
  flex-direction: column;
  gap: 24px;
`;

const StatusMessage = styled.div<{ tone?: 'default' | 'danger' }>`
  padding: 18px 20px;
  border-radius: ${({ theme }) => theme.radii.md};
  background: ${({ tone }) => (tone === 'danger' ? 'rgba(239, 68, 68, 0.16)' : 'rgba(59, 130, 246, 0.16)')};
  color: ${({ tone, theme }) => (tone === 'danger' ? theme.colors.danger : theme.colors.textPrimary)};
  border: 1px solid ${({ tone }) => (tone === 'danger' ? 'rgba(239, 68, 68, 0.32)' : 'rgba(59, 130, 246, 0.32)')};
`;

const FooterNote = styled.footer`
  margin-top: 12px;
  color: ${({ theme }) => theme.colors.textSecondary};
  font-size: 12px;

  code {
    background: rgba(15, 118, 110, 0.16);
    padding: 2px 6px;
    border-radius: ${({ theme }) => theme.radii.sm};
  }
`;

const FiltersSection = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: 16px;
  align-items: flex-start;
  justify-content: space-between;

  @media (max-width: 768px) {
    flex-direction: column;
    align-items: stretch;
  }
`;

const FilterSummary = styled.div`
  display: inline-flex;
  align-items: center;
  gap: 12px;
  padding: 12px 16px;
  border-radius: ${({ theme }) => theme.radii.md};
  border: 1px solid ${({ theme }) => theme.colors.border};
  background: rgba(148, 163, 184, 0.1);
  font-size: 13px;

  @media (max-width: 768px) {
    width: 100%;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 8px;
  }
`;

const SummaryItem = styled.div`
  display: flex;
  flex-direction: column;
  gap: 2px;

  label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: ${({ theme }) => theme.colors.textSecondary};
  }

  span {
    font-weight: 600;
  }
`;

const ClearFiltersButton = styled.button`
  border-radius: ${({ theme }) => theme.radii.sm};
  border: 1px solid ${({ theme }) => theme.colors.border};
  background: transparent;
  color: ${({ theme }) => theme.colors.textSecondary};
  padding: 6px 10px;
  font-size: 12px;
  cursor: pointer;

  &:hover {
    color: ${({ theme }) => theme.colors.textPrimary};
    border-color: ${({ theme }) => theme.colors.primary};
  }
`;
