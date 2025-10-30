#!/usr/bin/env python3
"""Inspect and manage the WhatsApp outbox before wiring Make/Twilio."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.loader import get_path


def iter_entries(path: Path) -> Iterable[dict]:
    if not path.exists():
        return []
    with path.open('r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Muestra o limpia el outbox de WhatsApp")
    parser.add_argument('--limit', type=int, default=20, help='Número de mensajes a mostrar (default: 20)')
    parser.add_argument('--clear', action='store_true', help='Vacía el archivo después de mostrarlo')
    args = parser.parse_args(argv)

    outbox_path = get_path('reports', 'whatsapp_outbox', default='reports/whatsapp_outbox.jsonl')
    outbox_path = Path(outbox_path)

    entries = list(iter_entries(outbox_path))
    if not entries:
        print(f"[whatsapp-outbox] No hay mensajes en {outbox_path}")
        if args.clear and outbox_path.exists():
            outbox_path.unlink()
            print("[whatsapp-outbox] Archivo eliminado para reiniciar la cola")
        return 0

    print(f"[whatsapp-outbox] Mostrando hasta {args.limit} mensajes de {len(entries)} totales\n")
    for entry in entries[: max(args.limit, 1)]:
        print("=" * 80)
        print(entry.get('message', '(sin mensaje)'))
        meta = {k: v for k, v in entry.items() if k not in {'message'}}
        print("-" * 80)
        print(json.dumps(meta, ensure_ascii=False, indent=2))

    if args.clear:
        outbox_path.unlink(missing_ok=True)
        print(f"[whatsapp-outbox] Cola limpiada ({outbox_path})")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
