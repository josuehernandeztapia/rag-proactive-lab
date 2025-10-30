"""Utilities to write WhatsApp-ready payloads to the shared outbox."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from config.loader import get_path

_OUTBOX_PATH: Path | None = None


def _resolve_outbox_path() -> Path:
    global _OUTBOX_PATH
    if _OUTBOX_PATH is None:
        path = get_path('reports', 'whatsapp_outbox', default='reports/whatsapp_outbox.jsonl')
        _OUTBOX_PATH = Path(path)
    return _OUTBOX_PATH


def append_message(
    *,
    agent: str,
    message: str,
    contact: str | None = None,
    placa: str | None = None,
    quick_replies: Iterable[str] | None = None,
    metadata: Mapping[str, Any] | None = None,
    channel: str = 'whatsapp',
) -> None:
    """Append a message to the shared WhatsApp outbox.

    Parameters
    ----------
    agent:
        Logical owner of the message (`pia`, `guardian`, `postventa`, etc.).
    message:
        Final text that should be sent to the customer.
    contact:
        Optional phone number or identifier the orchestration layer will use.
    placa:
        Plate/unit identifier for traceability.
    quick_replies:
        Suggested quick replies for the channel (converted to list).
    metadata:
        Extra context (strategy, severity, etc.). Must be JSON serialisable.
    channel:
        Channel name; defaults to `whatsapp` but kept configurable for future reuse.
    """
    if not message:
        return

    payload = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'agent': agent,
        'channel': channel,
        'message': message,
    }
    if contact:
        payload['contact'] = contact
    if placa:
        payload['placa'] = placa
    if quick_replies:
        payload['quick_replies'] = list(quick_replies)
    if metadata:
        try:
            payload['metadata'] = json.loads(json.dumps(metadata, ensure_ascii=False))
        except Exception:
            # Fallback: stringify non-serialisable objects
            payload['metadata'] = {k: str(v) for k, v in metadata.items()}

    outbox_path = _resolve_outbox_path()
    outbox_path.parent.mkdir(parents=True, exist_ok=True)
    with outbox_path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + '\n')


__all__ = ['append_message']
