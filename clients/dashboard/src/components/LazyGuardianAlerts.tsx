import { lazy, Suspense } from 'react';
import type { GuardianAlert } from '../types';
import { Card } from './Card';

const GuardianAlerts = lazy(() => import('./GuardianAlerts').then(module => ({ default: module.GuardianAlerts })));

interface LazyGuardianAlertsProps {
  alerts: GuardianAlert[];
}

const LoadingPlaceholder = () => (
  <Card title="Guardian de Flota" subtitle="Cargando alertas...">
    <div style={{
      padding: '32px 20px',
      textAlign: 'center',
      color: '#64748b',
      background: 'rgba(15, 23, 42, 0.4)',
      borderRadius: '8px',
      border: '1px dashed rgba(148, 163, 184, 0.24)'
    }}>
      <div>📡 Procesando alertas de Guardian...</div>
    </div>
  </Card>
);

export function LazyGuardianAlerts({ alerts }: LazyGuardianAlertsProps) {
  return (
    <Suspense fallback={<LoadingPlaceholder />}>
      <GuardianAlerts alerts={alerts} />
    </Suspense>
  );
}