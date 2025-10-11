#!/usr/bin/env python3
"""Validate PIA prompt templates can be rendered with synthetic data.

The script parses every template in `prompts/pia/`, builds a dummy context for
all placeholders detected via `str.format`, and renders the template. If any
placeholder is missing or a template contains malformed format fields, the
script exits with a non-zero status so CI/pre-commit hooks can catch it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from string import Formatter
from typing import Dict

ROOT = Path(__file__).resolve().parents[1]
PROMPTS_DIR = ROOT / "prompts" / "pia"
SUPPORTED_SUFFIXES = {".md", ".txt"}

# Pre-populate context with sensible defaults so the template keeps its shape.
DEFAULT_CONTEXT: Dict[str, object] = {
    "timestamp": "2024-10-06T00:00:00Z",
    "placa": "LAB-001",
    "market": "edomex",
    "plan_type": "proteccion_total",
    "plan_status": "vigente",
    "plan_valid_until": "2025-12-31",
    "plan_reset_cycle_days": 30,
    "protections_used": 1,
    "protections_allowed": 3,
    "protections_remaining": 2,
    "flag_summary": "sin banderas",
    "metric_snapshot": "sin métricas",
    "impact_projection": "sin impacto",
    "recommended_action": "Confirmar con el operador",
    "scenario_type": "DEFER",
    "scenario_summary": "DEFER, Δpago -1000, Δplazo +2",
    "critical_signals": "Sin banderas",
    "financial_snapshot": "- Pago esperado: $10,000.00",
    "action": "offer_protection",
    "outcome": "proposed_protection",
    "risk_band": "alto",
    "reason": "Consumo bajo",
    "channel": "whatsapp",
    "template": "PIA_RECORDATORIO",
    "notes": "sin notas",
    "reference_ts": "2024-10-06T00:00:00Z",
    "balance": "$120,000.00",
    "payment": "$8,500.00",
    "primary_scenario": "DEFER, Δpago -1000",
    "scenario_overview": "STEPDOWN, Δpago -500",
    "scenarios_list": "- DEFER\n- STEPDOWN",
    "max_tags": 5,
}


def _collect_placeholders(content: str) -> set[str]:
    names: set[str] = set()
    formatter = Formatter()
    for literal_text, field_name, _, _ in formatter.parse(content):
        if field_name is None or field_name == "":
            continue
        # ignore format spec/accesor such as "{foo!r}" or "{foo:.2f}" keeping the base key
        base_name = field_name.split(".")[0].split("[")[0]
        names.add(base_name)
    return names


def _build_context(placeholders: set[str]) -> Dict[str, object]:
    context = DEFAULT_CONTEXT.copy()
    for name in placeholders:
        context.setdefault(name, f"__{name}__")
    return context


def validate_templates(verbose: bool = False) -> int:
    if not PROMPTS_DIR.exists():
        print(f"Prompts directory not found: {PROMPTS_DIR}", file=sys.stderr)
        return 1

    failures = 0
    for path in sorted(PROMPTS_DIR.glob("**/*")):
        if path.name == "manifest.json" or not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            if suffix == ".json":
                # ensure JSON is valid even without placeholders
                try:
                    json.loads(path.read_text(encoding="utf-8"))
                except json.JSONDecodeError as exc:
                    print(f"✗ {path.relative_to(ROOT)}: invalid JSON ({exc})", file=sys.stderr)
                    failures += 1
            continue
        content = path.read_text(encoding="utf-8")
        placeholders = _collect_placeholders(content)
        context = _build_context(placeholders)
        try:
            rendered = content.format_map(context)
        except KeyError as exc:
            print(f"✗ {path.relative_to(ROOT)}: missing placeholder value {exc}", file=sys.stderr)
            failures += 1
            continue
        except ValueError as exc:
            print(f"✗ {path.relative_to(ROOT)}: format error ({exc})", file=sys.stderr)
            failures += 1
            continue
        if verbose:
            print(f"✓ {path.relative_to(ROOT)} ({len(rendered)} chars)")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate PIA prompt templates")
    parser.add_argument("--verbose", action="store_true", help="Show successful validations")
    args = parser.parse_args()
    failures = validate_templates(verbose=args.verbose)
    if failures:
        print(f"\n{failures} template(s) failed validation.", file=sys.stderr)
        return 1
    if args.verbose:
        print("All PIA templates rendered successfully.")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    sys.exit(main())
