import { Injectable, inject } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { BehaviorSubject, Observable, of, throwError } from 'rxjs';
import { catchError, delay, map, timeout } from 'rxjs/operators';

type EndpointConfig = { path: string; method: 'GET'|'POST'; timeout?: number };

@Injectable({ providedIn: 'root' })
export class ApiConfigService {
  private http = inject(HttpClient);

  private config = new BehaviorSubject<{ baseUrl: string } | null>({ baseUrl: (window as any).__BFF_BASE__ || 'http://localhost:3000' });
  config$ = this.config.asObservable();

  private readonly ENDPOINTS: Record<string, EndpointConfig> = {
    VOICE_ANALYZE: { path: '/v1/voice/analyze', method: 'POST', timeout: 15000 },
    VOICE_EVALUATE_AUDIO: { path: '/v1/voice/evaluate-audio', method: 'POST', timeout: 20000 },
    WHISPER_TRANSCRIBE: { path: '/v1/voice/whisper-transcribe', method: 'POST', timeout: 20000 }
  };

  private MOCK = {
    enabled: true,
    delay: 700,
    responses: {
      VOICE_EVALUATE_AUDIO: {
        transcript: 'mock transcript', words: ['mi','hijo','me','cubre'],
        latencyIndex: 0.15, pitchVar: 0.2, disfluencyRate: 0.05, energyStability: 0.85, honestyLexicon: 0.6,
        voiceScore: 0.82, flags: ['stableEnergy'], decision: 'GO'
      },
      VOICE_ANALYZE: {
        latencyIndex: 0.3, pitchVar: 0.4, disfluencyRate: 0.2, energyStability: 0.7, honestyLexicon: 0.5,
        voiceScore: 0.68, flags: ['unstablePitch'], decision: 'REVIEW'
      }
    } as Record<string, any>
  };

  getEndpointUrl(name: string): string {
    const cfg = this.config.value;
    const ep = this.ENDPOINTS[name];
    if (!cfg || !ep) throw new Error(`Endpoint ${name} not configured`);
    return `${cfg.baseUrl}${ep.path}`;
  }

  isMockMode() { return this.MOCK.enabled; }
  toggleMockMode(v: boolean) { this.MOCK.enabled = v; }

  request<T>(endpoint: string, body?: any, headers?: HttpHeaders): Observable<T> {
    const ep = this.ENDPOINTS[endpoint];
    if (!ep) return throwError(() => new Error(`Endpoint ${endpoint} not configured`));

    if (this.isMockMode()) {
      const mock = this.MOCK.responses[endpoint];
      return of(mock as T).pipe(delay(this.MOCK.delay));
    }

    const url = this.getEndpointUrl(endpoint);
    const req$ = ep.method === 'GET' ? this.http.get<T>(url, { headers }) : this.http.post<T>(url, body, { headers });
    return req$.pipe(timeout(ep.timeout || 15000));
  }
}

