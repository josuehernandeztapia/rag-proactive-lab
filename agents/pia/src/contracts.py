"""Synthetic loader for protection contract metadata."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_CONTRACTS_PATH = ROOT_DIR / "data" / "pia" / "protection_contracts_dummy.csv"


@dataclass(frozen=True)
class ProtectionContract:
    placa: str
    plan_type: str
    protections_allowed: int
    protections_used: int
    status: str = "active"
    valid_until: Optional[str] = None
    reset_cycle_days: Optional[int] = None
    requires_manual_review: bool = False


def _normalize_placa(value: str | None) -> str:
    return str(value or "").strip().upper()


@lru_cache(maxsize=None)
def _load_contracts(path: Path = DEFAULT_CONTRACTS_PATH) -> dict[str, ProtectionContract]:
    contracts: dict[str, ProtectionContract] = {}
    if not path.exists():
        return contracts
    try:
        with path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                placa = _normalize_placa(row.get("placa"))
                if not placa:
                    continue
                try:
                    allowed = int(row.get("protections_allowed", 0))
                except (TypeError, ValueError):
                    allowed = 0
                try:
                    used = int(row.get("protections_used", 0))
                except (TypeError, ValueError):
                    used = 0
                plan_type = str(row.get("plan_type") or "").strip() or "unknown"
                status = str(row.get("status") or "").strip() or "active"
                valid_until = str(row.get("valid_until") or "").strip() or None
                try:
                    reset_cycle_days = int(row.get("reset_cycle_days", 0) or 0)
                except (TypeError, ValueError):
                    reset_cycle_days = None
                requires_manual_review = str(row.get("requires_manual_review") or "").strip().lower() in {"1", "true", "yes", "y"}
                contracts[placa] = ProtectionContract(
                    placa=placa,
                    plan_type=plan_type,
                    protections_allowed=allowed,
                    protections_used=used,
                    status=status,
                    valid_until=valid_until,
                    reset_cycle_days=reset_cycle_days,
                    requires_manual_review=requires_manual_review,
                )
    except Exception:
        return contracts
    return contracts


def get_contract_for_placa(placa: str, *, path: Path = DEFAULT_CONTRACTS_PATH) -> Optional[ProtectionContract]:
    placa_key = _normalize_placa(placa)
    if not placa_key:
        return None
    return _load_contracts(path).get(placa_key)


__all__ = ["ProtectionContract", "get_contract_for_placa", "DEFAULT_CONTRACTS_PATH"]
