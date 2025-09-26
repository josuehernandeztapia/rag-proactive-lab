#!/usr/bin/env python3
"""Genera escenarios sintéticos adicionales para entrenamiento HASE."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import sys
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.hase.src.synthetic import SyntheticScenario, generate_synthetic_rows

DEFAULT_INPUT = ROOT / "data" / "hase" / "hase_training_dataset.csv.gz"
DEFAULT_OUTPUT = ROOT / "data" / "hase" / "hase_training_dataset_augmented.csv.gz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generar dataset HASE con escenarios sintéticos adicionales")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument(
        "--scenarios",
        type=str,
        default="consumption_gap:500,downtime_spike:400,fault_alert:400,stable_high:400,recovery:300",
        help="Lista de escenarios con formato nombre:count separados por coma",
    )
    return parser.parse_args()


def parse_scenarios(raw: str) -> dict[str, SyntheticScenario]:
    mapping = {}
    for item in raw.split(","):
        if not item:
            continue
        name, count_str = item.split(":", 1)
        count = int(count_str)
        if name == "consumption_gap":
            label_flag = 1
            reason = "synthetic_consumption_gap"
        elif name == "downtime_spike":
            label_flag = 1
            reason = "synthetic_downtime_spike"
        elif name == "fault_alert":
            label_flag = 1
            reason = "synthetic_fault_alert"
        elif name == "stable_high":
            label_flag = 0
            reason = "synthetic_stable_high"
        elif name == "recovery":
            label_flag = 0
            reason = "synthetic_recovery"
        else:
            label_flag = 1
            reason = f"synthetic_{name}"
        mapping[name] = SyntheticScenario(
            name=name,
            count=count,
            label_flag=label_flag,
            label_reason=reason,
        )
    return mapping


def main() -> int:
    args = parse_args()
    base = pd.read_csv(args.input)
    scenario_cfgs = parse_scenarios(args.scenarios)

    rng = np.random.default_rng(args.seed)
    synthetic = generate_synthetic_rows(base, scenario_cfgs, rng=rng)
    augmented = pd.concat([base, synthetic], ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    augmented.to_csv(args.output, index=False, compression="infer")
    print(
        f"Dataset original: {len(base)} filas | Sintético: {len(synthetic)} filas | Total: {len(augmented)}"
    )
    print(f"Guardado en {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
