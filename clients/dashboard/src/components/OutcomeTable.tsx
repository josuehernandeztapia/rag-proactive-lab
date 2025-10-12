import { useMemo, useState } from 'react';
import styled from 'styled-components';
import type { ColumnDef, SortingState } from '@tanstack/react-table';
import { flexRender, getCoreRowModel, getSortedRowModel, useReactTable } from '@tanstack/react-table';
import dayjs from 'dayjs';
import type { OutcomeScenarioSummary } from '../types';
import { Card } from './Card';
import { Badge } from './Badge';

interface OutcomeTableProps {
  data: OutcomeScenarioSummary[];
  selectedPlaca?: string;
  onSelectPlaca?: (placa: string) => void;
}

const TableWrapper = styled.div`
  overflow-x: auto;
`;

const Table = styled.table`
  width: 100%;
  border-collapse: collapse;
  font-size: 14px;

  th,
  td {
    padding: 12px 16px;
    border-bottom: 1px solid ${({ theme }) => theme.colors.border};
    white-space: nowrap;
    text-align: left;
  }

  th {
    font-weight: 600;
    color: ${({ theme }) => theme.colors.textSecondary};
    background: rgba(15, 118, 110, 0.08);
    text-transform: uppercase;
    letter-spacing: 0.04em;
    font-size: 12px;
    cursor: pointer;
  }

  tr:last-child td {
    border-bottom: none;
  }
`;

const Row = styled.tr<{ clickable: boolean; selected: boolean }>`
  background: ${({ selected }) => (selected ? 'rgba(59, 130, 246, 0.14)' : 'transparent')};
  cursor: ${({ clickable }) => (clickable ? 'pointer' : 'default')};

  &:hover {
    background: ${({ clickable }) => (clickable ? 'rgba(148, 163, 184, 0.12)' : 'inherit')};
  }
`;

const ManualBadge = styled(Badge)`
  font-size: 11px;
`;

export function OutcomeTable({ data, selectedPlaca, onSelectPlaca }: OutcomeTableProps) {
  const [sorting, setSorting] = useState<SortingState>([{ id: 'timestamp', desc: true }]);

  const columns = useMemo<ColumnDef<OutcomeScenarioSummary>[]>(
    () => [
      {
        accessorKey: 'timestamp',
        header: 'Fecha',
        cell: (info) => dayjs(String(info.getValue())).format('YYYY-MM-DD HH:mm'),
      },
      {
        accessorKey: 'placa',
        header: 'Placa',
      },
      {
        accessorKey: 'action',
        header: 'Acción',
      },
      {
        accessorKey: 'scenario',
        header: 'Escenario',
      },
      {
        accessorKey: 'annualIrr',
        header: 'TIR anual',
        cell: (info) => {
          const value = info.getValue<number | null>();
          return value !== null && Number.isFinite(value) ? `${(value * 100).toFixed(2)} %` : '—';
        },
      },
      {
        accessorKey: 'paymentChange',
        header: 'Δ Pago',
        cell: (info) => formatCurrency(info.getValue()),
      },
      {
        accessorKey: 'termChange',
        header: 'Δ Plazo',
        cell: (info) => {
          const value = info.getValue<number | null>();
          if (!value) return '—';
          const sign = value > 0 ? '+' : '';
          return `${sign}${value}m`;
        },
      },
      {
        accessorKey: 'requiresManualReview',
        header: 'Revisión manual',
        cell: (info) => (info.getValue<boolean>() ? <ManualBadge tone="warning">Sí</ManualBadge> : 'No'),
      },
    ],
    [],
  );

  const table = useReactTable({
    data,
    columns,
    state: { sorting },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
  });

  if (!data.length) {
    return (
      <Card title="TIR Drilldown" subtitle="Escenarios evaluados por PIA con impacto en pago y plazo.">
        <EmptyState>No hay escenarios recientes que cumplan con los filtros.</EmptyState>
      </Card>
    );
  }

  return (
    <Card title="TIR Drilldown" subtitle="Escenarios evaluados por PIA con impacto en pago y plazo.">
      <TableWrapper>
        <Table>
          <thead>
            {table.getHeaderGroups().map((headerGroup) => (
              <tr key={headerGroup.id}>
                {headerGroup.headers.map((header) => (
                  <th key={header.id} onClick={header.column.getToggleSortingHandler()}>
                    {flexRender(header.column.columnDef.header, header.getContext())}
                    {{ asc: ' ▲', desc: ' ▼' }[header.column.getIsSorted() as string] ?? null}
                  </th>
                ))}
              </tr>
            ))}
          </thead>
          <tbody>
            {table.getRowModel().rows.map((row) => (
              <Row
                key={row.id}
                clickable={Boolean(onSelectPlaca)}
                selected={Boolean(selectedPlaca && row.original.placa === selectedPlaca)}
                onClick={() => onSelectPlaca?.(row.original.placa)}
              >
                {row.getVisibleCells().map((cell) => (
                  <td key={cell.id}>{flexRender(cell.column.columnDef.cell, cell.getContext())}</td>
                ))}
              </Row>
            ))}
          </tbody>
        </Table>
      </TableWrapper>
    </Card>
  );
}

const EmptyState = styled.div`
  padding: 24px;
  border-radius: ${({ theme }) => theme.radii.md};
  border: 1px dashed ${({ theme }) => theme.colors.border};
  background: rgba(15, 118, 110, 0.08);
  color: ${({ theme }) => theme.colors.textSecondary};
  text-align: center;
`;

function formatCurrency(value: unknown): string {
  const number = typeof value === 'number' ? value : Number(value ?? 0);
  if (!Number.isFinite(number) || number === 0) return '—';
  const formatter = new Intl.NumberFormat('es-MX', {
    style: 'currency',
    currency: 'MXN',
    maximumFractionDigits: 0,
  });
  return formatter.format(number);
}
