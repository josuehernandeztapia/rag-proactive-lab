import styled from 'styled-components';
import type { PlanSummaryRow } from '../types';
import { Card } from './Card';
import { Badge } from './Badge';

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
    text-align: left;
    white-space: nowrap;
  }

  th {
    font-weight: 600;
    color: ${({ theme }) => theme.colors.textSecondary};
    background: rgba(15, 118, 110, 0.08);
    text-transform: uppercase;
    letter-spacing: 0.04em;
    font-size: 12px;
  }

  tr:last-child td {
    border-bottom: none;
  }
`;

const Highlight = styled.span<{ tone: 'warning' | 'danger' }>`
  color: ${({ tone, theme }) => (tone === 'warning' ? theme.colors.warning : theme.colors.danger)};
  font-weight: 600;
`;

interface ProtectionHeatmapProps {
  planSummary: PlanSummaryRow[];
}

export function ProtectionHeatmap({ planSummary }: ProtectionHeatmapProps) {
  return (
    <Card title="Protección por plan" subtitle="Promedios y banderas clave para cada tipo de plan.">
      <TableWrapper>
        <Table>
          <thead>
            <tr>
              <th>Plan</th>
              <th>Contratos</th>
              <th>Protecciones prom.</th>
              <th>Protecciones mediana</th>
              <th>Manual</th>
              <th>Expirados</th>
              <th>Protecciones negativas</th>
            </tr>
          </thead>
          <tbody>
            {planSummary.map((row) => (
              <tr key={row.plan_type}>
                <td>
                  <Badge tone="neutral">{row.plan_type}</Badge>
                </td>
                <td>{row.contratos.toLocaleString('es-MX')}</td>
                <td>{row.protecciones_restantes_promedio.toFixed(2)}</td>
                <td>{row.protecciones_restantes_mediana.toFixed(2)}</td>
                <td>{row.contratos_manual > 0 ? <Highlight tone="warning">{row.contratos_manual}</Highlight> : row.contratos_manual}</td>
                <td>{row.contratos_expirados > 0 ? <Highlight tone="danger">{row.contratos_expirados}</Highlight> : row.contratos_expirados}</td>
                <td>{row.contratos_negative > 0 ? <Highlight tone="danger">{row.contratos_negative}</Highlight> : row.contratos_negative}</td>
              </tr>
            ))}
          </tbody>
        </Table>
      </TableWrapper>
    </Card>
  );
}
