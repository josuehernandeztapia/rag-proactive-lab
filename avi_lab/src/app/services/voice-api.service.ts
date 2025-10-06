import { Injectable, inject } from '@angular/core';
import { HttpHeaders } from '@angular/common/http';
import { Observable } from 'rxjs';
import { ApiConfigService } from '../core/api-config.service';
import { VoiceAnalyzeRes, VoiceEvaluateRes } from '../core/avi-api-contracts';

@Injectable({ providedIn: 'root' })
export class VoiceApiService {
  private api = inject(ApiConfigService);

  evaluateAudio(file: File | Blob, questionId: string, contextId?: string): Observable<VoiceEvaluateRes> {
    const fd = new FormData();
    fd.append('file', file);
    if (questionId) fd.append('questionId', questionId);
    if (contextId) fd.append('contextId', contextId);
    return this.api.request<VoiceEvaluateRes>('VOICE_EVALUATE_AUDIO', fd);
  }

  analyzeFeatures(req: {
    latencySec: number; answerDurationSec: number;
    pitchSeriesHz: number[]; energySeries: number[]; words: string[];
  }): Observable<VoiceAnalyzeRes> {
    const headers = new HttpHeaders().set('Content-Type', 'application/json');
    return this.api.request<VoiceAnalyzeRes>('VOICE_ANALYZE', req, headers);
  }
}

