import { Component, inject } from '@angular/core';
import { RouterLink, RouterOutlet } from '@angular/router';
import { CommonModule } from '@angular/common';
import { ApiConfigService } from './core/api-config.service';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, RouterOutlet, RouterLink],
  template: `
  <div class="container">
    <h1>AVI Lab</h1>
    <p class="muted">Sandbox para pruebas de AVI</p>

    <div class="row" style="margin: 16px 0">
      <a routerLink="/runner" class="btn btn-primary">Runner</a>
      <a routerLink="/metrics" class="btn btn-ghost">Métricas</a>
      <button class="btn btn-ghost" (click)="toggleMock()">{{ mockLabel }}</button>
    </div>

    <router-outlet></router-outlet>
  </div>
  `
})
export class AppComponent {
  private api = inject(ApiConfigService);
  get mockLabel() { return this.api.isMockMode() ? 'Mock: ON' : 'Mock: OFF'; }
  toggleMock() { this.api.toggleMockMode(!this.api.isMockMode()); }
}

