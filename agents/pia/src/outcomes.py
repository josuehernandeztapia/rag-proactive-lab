"""Utilities to log PIA outcomes and derive aggregates for HASE."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd
import logging

from .rules import PIADecision
from .llm_service import feature_enabled, get_llm_service

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LOG_PATH = ROOT / "data" / "pia" / "pia_outcomes_log.csv"

LOGGER = logging.getLogger("pia.outcomes")


def _extract_behaviour_tags(meta: dict) -> list[str]:
    tags = meta.get("behaviour_tags") if isinstance(meta, dict) else []
    if isinstance(tags, str):
        tags = [tags]
    if not isinstance(tags, list):
        return []
    return [str(tag).strip() for tag in tags if str(tag).strip()]


def _extract_behaviour_notes(meta: dict) -> list[str]:
    notes = meta.get("behaviour_notes") if isinstance(meta, dict) else []
    if isinstance(notes, str):
        notes = [notes]
    if not isinstance(notes, list):
        return []
    return [str(note).strip() for note in notes if str(note).strip()]


@dataclass(frozen=True)
class OutcomeRecord:
    timestamp: datetime
    placa: str
    plaza: str
    action: str
    outcome: str
    reason: str
    template: str
    risk_band: str
    scenario: Optional[str]
    notes: str
    details: dict[str, Any]
    metadata: dict[str, Any]
    plan_type: Optional[str]
    protections_used: Optional[int]
    protections_allowed: Optional[int]


def _ensure_log_header(path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "timestamp",
                "placa",
                "plaza",
                "action",
                "outcome",
                "reason",
                "template",
                "risk_band",
                "scenario",
                "notes",
                "details_json",
                "metadata_json",
                "plan_type",
                "protections_used",
                "protections_allowed",
                "plan_status",
                "plan_valid_until",
                "plan_reset_cycle_days",
                "plan_requires_manual_review",
            ],
        )
        writer.writeheader()


def record_outcome(
    decision: PIADecision,
    outcome: str,
    *,
    plaza: str | None = None,
    notes: str = "",
    metadata: Optional[dict[str, Any]] = None,
    log_path: Path = DEFAULT_LOG_PATH,
) -> OutcomeRecord:
    """Persist an outcome for a given decision."""
    record = OutcomeRecord(
        timestamp=datetime.now(timezone.utc),
        placa=str(decision.placa),
        plaza=plaza or str(decision.details.get("plaza") or ""),
        action=decision.action,
        outcome=outcome,
        reason=decision.reason,
        template=decision.template,
        risk_band=decision.risk_band,
        scenario=decision.scenario,
        notes=notes,
        details=decision.details,
        metadata=metadata or {},
        plan_type=metadata.get("protection_plan", {}).get("plan_type") if metadata else None,
        protections_used=metadata.get("protection_plan", {}).get("protections_used") if metadata else None,
        protections_allowed=metadata.get("protection_plan", {}).get("protections_allowed") if metadata else None,
    )
    _ensure_log_header(log_path)
    row = {
        "timestamp": record.timestamp.isoformat(),
        "placa": record.placa,
        "plaza": record.plaza,
        "action": record.action,
        "outcome": record.outcome,
        "reason": record.reason,
        "template": record.template,
        "risk_band": record.risk_band,
        "scenario": record.scenario or "",
        "notes": notes,
        "details_json": json.dumps(record.details, ensure_ascii=False, default=str),
        "metadata_json": json.dumps(record.metadata, ensure_ascii=False, default=str),
        "plan_type": record.plan_type or "",
        "protections_used": record.protections_used if record.protections_used is not None else "",
        "protections_allowed": record.protections_allowed if record.protections_allowed is not None else "",
        "plan_status": record.metadata.get("protection_plan", {}).get("status") if record.metadata else "",
        "plan_valid_until": record.metadata.get("protection_plan", {}).get("valid_until") if record.metadata else "",
        "plan_reset_cycle_days": record.metadata.get("protection_plan", {}).get("reset_cycle_days") if record.metadata else "",
        "plan_requires_manual_review": record.metadata.get("protection_plan", {}).get("requires_manual_review") if record.metadata else "",
    }
    with log_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=row.keys())
        writer.writerow(row)
    _maybe_generate_case_note(record)
    return record


def load_outcomes(log_path: Path = DEFAULT_LOG_PATH) -> pd.DataFrame:
    if not log_path.exists():
        raise FileNotFoundError(f"Outcome log not found: {log_path}")
    df = pd.read_csv(log_path)
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df["details_json"] = df["details_json"].apply(lambda value: json.loads(value) if isinstance(value, str) and value else {})
    df["metadata_json"] = df["metadata_json"].apply(lambda value: json.loads(value) if isinstance(value, str) and value else {})
    df["behaviour_tags"] = df["metadata_json"].apply(_extract_behaviour_tags)
    df["behaviour_notes"] = df["metadata_json"].apply(_extract_behaviour_notes)
    return df


def aggregate_outcomes(
    df: pd.DataFrame,
    *,
    windows: Iterable[int] = (30, 90, 180),
    reference: datetime | None = None,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    reference = reference or datetime.now(timezone.utc)
    reference_ts = pd.Timestamp(reference).tz_convert("UTC")
    df = df.copy()
    if "plan_type" not in df.columns:
        df["plan_type"] = pd.NA
    if "plan_status" not in df.columns:
        df["plan_status"] = pd.NA
    if "plan_valid_until" not in df.columns:
        df["plan_valid_until"] = pd.NA
    if "plan_reset_cycle_days" not in df.columns:
        df["plan_reset_cycle_days"] = pd.NA
    if "protections_used" not in df.columns:
        df["protections_used"] = pd.NA
    if "plan_requires_manual_review" not in df.columns:
        df["plan_requires_manual_review"] = pd.NA
    if "protections_allowed" not in df.columns:
        df["protections_allowed"] = pd.NA
    df["plan_type"] = df["plan_type"].replace("", pd.NA)
    df["plan_status"] = df["plan_status"].replace("", pd.NA)
    df["plan_valid_until"] = df["plan_valid_until"].replace("", pd.NA)
    df["plan_requires_manual_review"] = df["plan_requires_manual_review"].replace("", pd.NA)
    for column in ("protections_used", "protections_allowed"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["plan_reset_cycle_days"] = pd.to_numeric(df["plan_reset_cycle_days"], errors="coerce")
    df["plan_requires_manual_review"] = df["plan_requires_manual_review"].astype("boolean")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp", "placa"])
    df["timestamp"] = df["timestamp"].dt.tz_convert("UTC")
    df["count"] = 1

    base = (
        df.groupby("placa")
        .agg(
            outcomes_total=("count", "sum"),
            last_outcome_at=("timestamp", "max"),
        )
        .reset_index()
    )
    base["days_since_last_outcome"] = (reference_ts - base["last_outcome_at"]).dt.days
    base["last_outcome_at"] = base["last_outcome_at"].dt.tz_convert("UTC").dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    latest = (
        df.sort_values("timestamp")
        .groupby("placa")
        .agg(
            last_plan_type=("plan_type", "last"),
            last_protections_used=("protections_used", "last"),
            last_protections_allowed=("protections_allowed", "last"),
            last_plan_status=("plan_status", "last"),
            last_plan_valid_until=("plan_valid_until", "last"),
            last_plan_reset_cycle_days=("plan_reset_cycle_days", "last"),
            last_plan_requires_manual_review=("plan_requires_manual_review", "last"),
        )
        .reset_index()
    )
    latest["last_plan_type"] = latest["last_plan_type"].astype("string")
    latest["protections_remaining"] = (
        latest["last_protections_allowed"] - latest["last_protections_used"]
    )

    def _pivot(input_df: pd.DataFrame, suffix: str) -> pd.DataFrame:
        if input_df.empty:
            return pd.DataFrame()
        pivot = (
            input_df.pivot_table(
                index="placa",
                columns=["action", "outcome"],
                values="count",
                aggfunc="sum",
                fill_value=0,
            )
        )
        pivot.columns = [f"{action}_{outcome}_{suffix}" for action, outcome in pivot.columns]
        pivot = pivot.reset_index()
        return pivot

    aggregates = [base, latest]
    aggregates.append(_pivot(df, "all"))

    for window in windows:
        threshold = reference_ts - pd.Timedelta(days=window)
        window_df = df[df["timestamp"] >= threshold]
        pivot = _pivot(window_df, f"{window}d")
        if not pivot.empty:
            aggregates.append(pivot)

    result = aggregates[0]
    for pivot in aggregates[1:]:
        if pivot.empty:
            continue
        result = result.merge(pivot, on="placa", how="left")
    numeric_cols = result.select_dtypes(include=["number"]).columns
    if len(numeric_cols) > 0:
        result[numeric_cols] = result[numeric_cols].fillna(0)
    if "last_plan_type" in result.columns:
        result["last_plan_type"] = (
            result["last_plan_type"].replace({0: pd.NA, 0.0: pd.NA}).fillna("unknown")
        )
    if "last_plan_status" in result.columns:
        result["last_plan_status"] = (
            result["last_plan_status"].replace({0: pd.NA, 0.0: pd.NA}).fillna("unknown")
        )
    if "last_plan_valid_until" in result.columns:
        result["last_plan_valid_until"] = result["last_plan_valid_until"].fillna("")
    if "last_plan_reset_cycle_days" in result.columns:
        result["last_plan_reset_cycle_days"] = result["last_plan_reset_cycle_days"].fillna(0)
    if "last_plan_requires_manual_review" in result.columns:
        result["last_plan_requires_manual_review"] = result["last_plan_requires_manual_review"].fillna(False)

    # Behaviour signals (tags/notes) -> features
    if "behaviour_tags" in df.columns:
        exploded = (
            df[["placa", "behaviour_tags"]]
            .explode("behaviour_tags")
            .dropna(subset=["behaviour_tags"])
        )
        if not exploded.empty:
            pivot_behaviour = (
                exploded.assign(count=1)
                .pivot_table(
                    index="placa",
                    columns="behaviour_tags",
                    values="count",
                    aggfunc="sum",
                    fill_value=0,
                )
            )
            pivot_behaviour.columns = [
                f"behaviour_tag_{col}_count" for col in pivot_behaviour.columns
            ]
            pivot_behaviour = pivot_behaviour.reset_index()
            result = result.merge(pivot_behaviour, on="placa", how="left")

        def _last_non_empty(series: pd.Series) -> list[str]:
            for item in reversed(series.tolist()):
                if isinstance(item, list) and item:
                    return item
            return []

        latest_behaviour = (
            df.sort_values("timestamp")
            .groupby("placa")["behaviour_tags"]
            .agg(_last_non_empty)
            .reset_index()
        )
        if not latest_behaviour.empty:
            latest_behaviour["last_behaviour_tags"] = latest_behaviour["behaviour_tags"].apply(
                lambda tags: "|".join(tags) if tags else ""
            )
            latest_behaviour = latest_behaviour.drop(columns=["behaviour_tags"])
            result = result.merge(latest_behaviour, on="placa", how="left")

    if "behaviour_notes" in df.columns:
        latest_notes = (
            df.sort_values("timestamp")
            .groupby("placa")["behaviour_notes"]
            .agg(lambda notes_list: next((n for n in reversed(notes_list.tolist()) if n), []))
            .reset_index()
        )
        if not latest_notes.empty:
            latest_notes["last_behaviour_notes"] = latest_notes["behaviour_notes"].apply(
                lambda notes: "|".join(notes) if notes else ""
            )
            latest_notes = latest_notes.drop(columns=["behaviour_notes"])
            result = result.merge(latest_notes, on="placa", how="left")

    if "last_behaviour_tags" in result.columns:
        result["last_behaviour_tags"] = result["last_behaviour_tags"].fillna("")
    if "last_behaviour_notes" in result.columns:
        result["last_behaviour_notes"] = result["last_behaviour_notes"].fillna("")

    return result


__all__ = ["OutcomeRecord", "record_outcome", "load_outcomes", "aggregate_outcomes", "DEFAULT_LOG_PATH"]


def _maybe_generate_case_note(record: OutcomeRecord) -> None:
    """Trigger LLM-backed case note generation when enabled."""

    if not feature_enabled("case_notes"):
        return
    service = get_llm_service()
    if service is None:
        LOGGER.warning("PIA_LLM_CASE_NOTES habilitado pero PIA_LLM_MODE=disabled o llm_service no se inicializó")
        return
    payload = {
        "timestamp": record.timestamp.isoformat(),
        "placa": record.placa,
        "plaza": record.plaza,
        "action": record.action,
        "outcome": record.outcome,
        "reason": record.reason,
        "risk_band": record.risk_band,
        "template": record.template,
        "scenario": record.scenario,
        "notes": record.notes,
        "details": record.details,
        "metadata": record.metadata,
        "plan_type": record.plan_type,
        "protections_used": record.protections_used,
        "protections_allowed": record.protections_allowed,
    }
    try:
        result = service.render_case_note(payload)
        if not result:
            return
        service.persist_case_note(payload, result)
    except Exception as exc:  # pragma: no cover - defensive guard
        LOGGER.warning("No se pudo generar nota LLM: %s", exc)
