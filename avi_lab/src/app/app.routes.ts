import { Routes } from '@angular/router';

export const routes: Routes = [
  {
    path: 'runner',
    loadComponent: () => import('./pages/runner.component').then(m => m.RunnerComponent)
  },
  {
    path: 'metrics',
    loadComponent: () => import('./pages/metrics.component').then(m => m.MetricsComponent)
  },
  { path: '', pathMatch: 'full', redirectTo: 'runner' },
  { path: '**', redirectTo: 'runner' }
];

