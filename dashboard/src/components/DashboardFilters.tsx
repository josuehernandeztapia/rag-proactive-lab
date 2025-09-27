import type { ChangeEvent } from 'react';
import styled from 'styled-components';

const FiltersWrapper = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: 16px;
  align-items: flex-end;
`;

const Group = styled.div`
  display: flex;
  flex-direction: column;
  gap: 6px;

  label {
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: ${({ theme }) => theme.colors.textSecondary};
  }

  select,
  input {
    min-width: 200px;
    border-radius: ${({ theme }) => theme.radii.sm};
    border: 1px solid ${({ theme }) => theme.colors.border};
    background: ${({ theme }) => theme.colors.surface};
    color: ${({ theme }) => theme.colors.textPrimary};
    padding: 10px 12px;
    font-size: 14px;
  }
`;

const ClearButton = styled.button`
  border-radius: ${({ theme }) => theme.radii.sm};
  border: 1px solid ${({ theme }) => theme.colors.border};
  background: transparent;
  color: ${({ theme }) => theme.colors.textSecondary};
  padding: 10px 16px;
  font-size: 14px;
  cursor: pointer;
  transition: background 0.2s ease, border 0.2s ease;

  &:hover:not(:disabled) {
    background: rgba(148, 163, 184, 0.12);
    border-color: ${({ theme }) => theme.colors.primary};
  }

  &:disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }
`;

interface Option {
  value: string;
  label: string;
}

interface DashboardFiltersProps {
  scenarioOptions: Option[];
  plateOptions: Option[];
  plazaOptions: Option[];
  scenarioValue: string;
  plateValue: string;
  plazaValue: string;
  startDate: string;
  endDate: string;
  onScenarioChange: (value: string) => void;
  onPlateChange: (value: string) => void;
  onPlazaChange: (value: string) => void;
  onDateRangeChange: (field: 'start' | 'end', value: string) => void;
  onClear: () => void;
}

export function DashboardFilters({
  scenarioOptions,
  plateOptions,
  plazaOptions,
  scenarioValue,
  plateValue,
  plazaValue,
  startDate,
  endDate,
  onScenarioChange,
  onPlateChange,
  onPlazaChange,
  onDateRangeChange,
  onClear,
}: DashboardFiltersProps) {
  const handleScenario = (event: ChangeEvent<HTMLSelectElement>) => {
    onScenarioChange(event.target.value);
  };

  const handlePlate = (event: ChangeEvent<HTMLSelectElement>) => {
    onPlateChange(event.target.value);
  };

  const handlePlaza = (event: ChangeEvent<HTMLSelectElement>) => {
    onPlazaChange(event.target.value);
  };

  const isDisabled =
    scenarioValue === 'all' &&
    plateValue === 'all' &&
    plazaValue === 'all' &&
    !startDate &&
    !endDate;

  return (
    <FiltersWrapper>
      <Group>
        <label htmlFor="scenario-filter">Escenario</label>
        <select id="scenario-filter" value={scenarioValue} onChange={handleScenario}>
          {scenarioOptions.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
      </Group>
      <Group>
        <label htmlFor="plate-filter">Placa</label>
        <select id="plate-filter" value={plateValue} onChange={handlePlate}>
          {plateOptions.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
      </Group>
      <Group>
        <label htmlFor="plaza-filter">Plaza</label>
        <select id="plaza-filter" value={plazaValue} onChange={handlePlaza}>
          {plazaOptions.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
      </Group>
      <Group>
        <label htmlFor="start-date">Desde</label>
        <input
          id="start-date"
          type="date"
          value={startDate}
          max={endDate || undefined}
          onChange={(event) => onDateRangeChange('start', event.target.value)}
        />
      </Group>
      <Group>
        <label htmlFor="end-date">Hasta</label>
        <input
          id="end-date"
          type="date"
          value={endDate}
          min={startDate || undefined}
          onChange={(event) => onDateRangeChange('end', event.target.value)}
        />
      </Group>
      <ClearButton type="button" onClick={onClear} disabled={isDisabled}>
        Limpiar filtros
      </ClearButton>
    </FiltersWrapper>
  );
}
