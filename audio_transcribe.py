import os
from typing import Optional, Tuple

def _twilio_creds_from_legacy_file() -> Tuple[Optional[str], Optional[str]]:
    paths = [
        os.path.join(os.path.dirname(__file__), 'secrets.local.txt'),
        'secrets.local.txt',
        os.path.join(os.path.dirname(__file__), 'secrets.loca.txt'),
        'secrets.loca.txt',
    ]
    sid = None
    token = None
    try:
        for p in paths:
            if not os.path.exists(p):
                continue
            with open(p, 'r', encoding='utf-8', errors='ignore') as f:
                lines = [ln.strip() for ln in f.readlines()]
            for i, ln in enumerate(lines):
                low = ln.lower()
                if ('account sid' in low or low == 'twilio sid' or low == 'twilio') and i + 1 < len(lines):
                    j = i + 1
                    while j < len(lines) and not lines[j]:
                        j += 1
                    if j < len(lines) and lines[j].startswith('AC'):
                        sid = lines[j]
                if ('auth token' in low or low == 'twilio auth token') and i + 1 < len(lines):
                    j = i + 1
                    while j < len(lines) and not lines[j]:
                        j += 1
                    if j < len(lines) and len(lines[j]) >= 20:
                        token = lines[j]
        return sid, token
    except Exception:
        return None, None

def _fetch_bytes(url: str) -> Optional[bytes]:
    try:
        import httpx
    except Exception:
        return None
    auth = None
    sid = os.getenv('TWILIO_SID') or os.getenv('TWILIO_ACCOUNT_SID')
    token = os.getenv('TWILIO_AUTH_TOKEN')
    if not (sid and token):
        fsid, ftoken = _twilio_creds_from_legacy_file()
        sid = sid or fsid
        token = token or ftoken
    if sid and token:
        auth = (sid, token)
    try:
        with httpx.Client(timeout=20.0, follow_redirects=True) as cli:
            r = cli.get(url, auth=auth)
            r.raise_for_status()
            return r.content
    except Exception:
        try:
            import storage as _st  # type: ignore
            _st.log_event('media_fetch_failed', {'url': url, 'media_type': 'audio'})
        except Exception:
            pass
        return None


def transcribe_audio_from_url(url: str) -> Optional[str]:
    """Descarga audio desde URL (Twilio media) y transcribe con OpenAI Whisper.
    Devuelve texto o None si falla.
    """
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        try:
            import storage as _st  # type: ignore
            _st.log_event('asr_error', {'url': url, 'reason': 'missing_openai_api_key'})
        except Exception:
            pass
        return None
    data = _fetch_bytes(url)
    if not data:
        return None
    try:
        from openai import OpenAI
        import io
        client = OpenAI(api_key=api_key)
        bio = io.BytesIO(data)
        bio.name = 'audio.ogg'
        # Sugerir lenguaje y contexto para mejorar precisión en español técnico
        asr_model = os.getenv('ASR_MODEL', 'whisper-1').strip() or 'whisper-1'
        out = client.audio.transcriptions.create(
            model=asr_model,
            file=bio,
            language='es',
            temperature=0,
            prompt='Contexto: diagnóstico mecánico postventa Higer (español). Menciona síntomas, testigos, frenos, aceite, temperatura, humo.'
        )
        # SDK devuelve objeto; tomar texto
        text = getattr(out, 'text', None)
        if isinstance(text, str) and text.strip():
            return text.strip()
        # fallback
        try:
            return str(out)
        except Exception:
            return None
    except Exception as e:
        try:
            import storage as _st  # type: ignore
            _st.log_event('asr_error', {'url': url, 'reason': 'openai_whisper_exception', 'error': str(e)})
        except Exception:
            pass
        return None
