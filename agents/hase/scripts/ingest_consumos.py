#!/usr/bin/env python3
"""Integrar históricos de consumo GNV (AGS / Edomex) con features diarios."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE = ROOT.parent / "1.-Consumos GNV Aguascalientes y Edomex"
DEFAULT_OUTPUT = ROOT / "data" / "hase" / "consumos_unificados.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unificar datos de consumo GNV de AGS y Edomex")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_file(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8", compression="infer")


def main() -> int:
    args = parse_args()
    ags_path = args.source / "consumos_consolidados_unificado.ags.csv.gz"
    edo_path = args.source / "consumos_consolidados_unificado.edomex.csv.gz"
    frames = []
    for path in (ags_path, edo_path):
        if path.exists():
            frames.append(load_file(path))
    if not frames:
        raise SystemExit("No se encontraron archivos unificados de consumo")
    df = pd.concat(frames, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Consumo consolidado guardado en {args.output} ({len(df)} filas)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
