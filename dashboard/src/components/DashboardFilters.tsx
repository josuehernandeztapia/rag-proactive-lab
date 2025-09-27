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

  select {
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
  scenarioValue: string;
  plateValue: string;
  onScenarioChange: (value: string) => void;
  onPlateChange: (value: string) => void;
  onClear: () => void;
}

export function DashboardFilters({
  scenarioOptions,
  plateOptions,
  scenarioValue,
  plateValue,
  onScenarioChange,
  onPlateChange,
  onClear,
}: DashboardFiltersProps) {
  const handleScenario = (event: ChangeEvent<HTMLSelectElement>) => {
    onScenarioChange(event.target.value);
  };

  const handlePlate = (event: ChangeEvent<HTMLSelectElement>) => {
    onPlateChange(event.target.value);
  };

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
      <ClearButton type="button" onClick={onClear} disabled={scenarioValue === 'all' && plateValue === 'all'}>
        Limpiar filtros
      </ClearButton>
    </FiltersWrapper>
  );
}
