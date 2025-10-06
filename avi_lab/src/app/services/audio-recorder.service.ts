import { Injectable } from '@angular/core';

@Injectable({ providedIn: 'root' })
export class AudioRecorderService {
  private recorder?: MediaRecorder;
  private chunks: BlobPart[] = [];

  async start(): Promise<void> {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const mime = MediaRecorder.isTypeSupported('audio/webm') ? 'audio/webm' : 'audio/ogg';
    this.recorder = new MediaRecorder(stream, { mimeType: mime });
    this.chunks = [];
    this.recorder.ondataavailable = e => this.chunks.push(e.data);
    this.recorder.start();
  }

  async stop(): Promise<Blob> {
    return new Promise(resolve => {
      if (!this.recorder) return resolve(new Blob());
      this.recorder.onstop = () => resolve(new Blob(this.chunks, { type: this.recorder!.mimeType }));
      this.recorder.stop();
      this.recorder.stream.getTracks().forEach(t => t.stop());
    });
  }
}

