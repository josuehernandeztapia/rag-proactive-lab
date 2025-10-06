import { Component, inject, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ApiConfigService } from '../core/api-config.service';

@Component({
  standalone: true,
  selector: 'app-metrics',
  imports: [CommonModule],
  template: `
  <div class="card">
    <h2>Métricas API</h2>
    <p class="muted">Mock mode: {{ api.isMockMode() ? 'ON' : 'OFF' }}</p>
    <p class="muted">Base URL: {{ baseUrl }}</p>
    <p class="muted">Usa el Runner para generar llamadas.</p>
  </div>
  `
})
export class MetricsComponent {
  api = inject(ApiConfigService);
  baseUrl = (window as any).__BFF_BASE__ || 'http://localhost:3000';
}

