"""Utilities to orchestrate LLM interactions for PIA (alerts, notas, resúmenes)."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from langchain.schema import HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI
except ModuleNotFoundError:  # pragma: no cover - langchain optional in some envs
    HumanMessage = None  # type: ignore
    SystemMessage = None  # type: ignore
    ChatOpenAI = None  # type: ignore

ROOT_DIR = Path(__file__).resolve().parents[3]
PROMPTS_DIR = ROOT_DIR / "prompts" / "llm"
REPORTS_DIR = ROOT_DIR / "reports"
CASE_NOTES_DIR = REPORTS_DIR / "pia_case_notes"
CASE_NOTES_INDEX = REPORTS_DIR / "pia_llm_case_notes.jsonl"
ALERTS_INDEX = REPORTS_DIR / "pia_llm_alerts.jsonl"

_TRUE_VALUES = {"1", "true", "yes", "y", "on"}
_TEMPLATE_CACHE: Dict[str, str] = {}
_LLM_SERVICE: Optional["LLMService"] = None

_FEATURE_FLAGS = {
    "case_notes": ("PIA_LLM_CASE_NOTES", False),
    "alerts": ("PIA_LLM_ALERTS", False),
    "summaries": ("PIA_LLM_SUMMARIES", False),
    "behaviour": ("PIA_LLM_BEHAVIOUR", False),
}

LOGGER = logging.getLogger("pia.llm")


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in _TRUE_VALUES


def feature_enabled(feature: str, default: bool = False) -> bool:
    if feature in _FEATURE_FLAGS:
        env_name, fallback = _FEATURE_FLAGS[feature]
        return _env_flag(env_name, fallback)
    env_name = feature.upper()
    return _env_flag(env_name, default)


def _load_template(filename: str) -> str:
    if filename not in _TEMPLATE_CACHE:
        path = PROMPTS_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Template not found: {path}")
        _TEMPLATE_CACHE[filename] = path.read_text(encoding="utf-8").strip()
    return _TEMPLATE_CACHE[filename]


def _safe(value: Any, default: str = "-") -> str:
    if value is None:
        return default
    if isinstance(value, bool):
        return "sí" if value else "no"
    if isinstance(value, float):
        if value != value:  # NaN
            return default
        return f"{value:.2f}" if abs(value) < 1000 else f"{value:,.2f}"
    return str(value)


def _format_currency(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    return f"${number:,.2f}"


def _format_percent(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    return f"{number * 100:.1f}%"


def _parse_ts(raw: Optional[str]) -> Optional[datetime]:
    if not raw:
        return None
    value = raw.strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _slugify(text: str) -> str:
    cleaned = []
    for char in text:
        if char.isalnum():
            cleaned.append(char.lower())
        elif char in ("-", "_"):
            cleaned.append(char)
        else:
            cleaned.append("-")
    slug = "".join(cleaned).strip("-")
    return slug or "sin-placa"


@dataclass(frozen=True)
class LLMConfig:
    mode: str
    model: str
    temperature: float
    max_tokens: int
    timeout: int

    @classmethod
    def from_env(cls) -> "LLMConfig":
        mode = os.getenv("PIA_LLM_MODE", "template").strip().lower()
        model = os.getenv("PIA_LLM_MODEL", os.getenv("LLM_MODEL", "gpt-4o-mini")).strip()
        try:
            temperature = float(os.getenv("PIA_LLM_TEMPERATURE", "0.2"))
        except ValueError:
            temperature = 0.2
        try:
            max_tokens = int(os.getenv("PIA_LLM_MAX_TOKENS", "512"))
        except ValueError:
            max_tokens = 512
        try:
            timeout = int(os.getenv("PIA_LLM_TIMEOUT", "30"))
        except ValueError:
            timeout = 30
        return cls(mode=mode, model=model, temperature=temperature, max_tokens=max_tokens, timeout=timeout)


class LLMService:
    """Thin orchestrator for narrative generation and alerts."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.mode = config.mode
        self.logger = LOGGER
        self.case_notes_dir = CASE_NOTES_DIR
        self.case_notes_dir.mkdir(parents=True, exist_ok=True)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        self.client: Optional[ChatOpenAI] = None
        if self.mode == "openai":
            if ChatOpenAI is None or HumanMessage is None or SystemMessage is None:  # pragma: no cover - import guard
                raise RuntimeError("langchain_openai is required for openai mode")
            self.client = ChatOpenAI(
                model=config.model,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                timeout=config.timeout,
            )
        self.system_prompts = {
            "case_note": "Eres analista de riesgo para flotas. Explica decisiones con claridad y tono profesional.",
            "alert": "Eres un asistente proactivo de cobranza. Señala riesgos, urgencia y pasos accionables en menos de cinco frases.",
            "summary": "Eres asesor de protección. Explica opciones financieras en tono cercano y accionable.",
            "behaviour": "Eres analista que detecta señales de riesgo en transcripciones de operadores y clientes.",
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def render_case_note(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if self.mode == "disabled":
            return None
        context = self._prepare_case_note_context(payload)
        output = self._dispatch("case_note", context)
        if not output:
            return None
        return {"content": output.strip(), "context": context}

    def persist_case_note(self, payload: Dict[str, Any], result: Dict[str, Any]) -> Path:
        content = result.get("content")
        context = result.get("context", {})
        timestamp = _parse_ts(context.get("timestamp")) or datetime.now(timezone.utc)
        slug = _slugify(str(payload.get("placa") or context.get("placa") or "sin-placa"))
        filename = f"{slug}_{timestamp.strftime('%Y%m%dT%H%M%SZ')}.md"
        path = self.case_notes_dir / filename
        header = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "placa": slug,
            "prompt_version": context.get("prompt_version"),
            "mode": self.mode,
        }
        serialized_header = json.dumps(header, ensure_ascii=False, sort_keys=True)
        note_body = content if content.endswith("\n") else f"{content}\n"
        path.write_text(f"<!-- {serialized_header} -->\n{note_body}", encoding="utf-8")
        self._append_jsonl(CASE_NOTES_INDEX, {
            "generated_at": header["generated_at"],
            "placa": slug,
            "file": str(path.relative_to(ROOT_DIR)),
            "prompt_version": header["prompt_version"],
            "mode": self.mode,
        })
        return path

    def render_alert(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if self.mode == "disabled":
            return None
        context = self._prepare_alert_context(payload)
        output = self._dispatch("alert", context)
        if not output:
            return None
        return {"content": output.strip(), "context": context}

    def persist_alert(self, payload: Dict[str, Any], result: Dict[str, Any]) -> None:
        summary = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "placa": payload.get("placa"),
            "flags": result.get("context", {}).get("flag_summary"),
            "mode": self.mode,
            "prompt_version": result.get("context", {}).get("prompt_version"),
            "content": result.get("content"),
        }
        self._append_jsonl(ALERTS_INDEX, summary)

    def render_protection_summary(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if self.mode == "disabled":
            return None
        context = self._prepare_summary_context(payload)
        output = self._dispatch("summary", context)
        if not output:
            return None
        return {"content": output.strip(), "context": context}

    def extract_behaviour_signals(
        self,
        transcript: str,
        *,
        max_tags: int = 5,
    ) -> Optional[Dict[str, Any]]:
        text = (transcript or "").strip()
        if not text or self.mode == "disabled":
            return None

        if self.mode == "template":
            return self._heuristic_behaviour(text, max_tags=max_tags)

        context = {
            "transcript": text,
            "max_tags": max(1, max_tags),
            "prompt_version": "behaviour_v1",
        }
        user_prompt = self._render_template("behaviour_extract_prompt.txt", context)
        response = self._call_openai(
            self.system_prompts.get("behaviour", ""),
            user_prompt,
        )
        if response:
            parsed = self._parse_behaviour_response(response)
            if parsed:
                return self._limit_behaviour(parsed, max_tags=max_tags)
        return self._heuristic_behaviour(text, max_tags=max_tags)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _dispatch(self, kind: str, context: Dict[str, Any]) -> Optional[str]:
        template_map = {
            "case_note": ("case_note_template.md", "case_note_prompt.txt"),
            "alert": ("alert_template.md", "alert_prompt.txt"),
            "summary": ("protection_summary_template.md", "protection_summary_prompt.txt"),
        }
        template_name, prompt_name = template_map[kind]
        if self.mode == "template":
            return self._render_template(template_name, context)
        if self.mode == "openai":
            user_prompt = self._render_template(prompt_name, context)
            system_prompt = self.system_prompts.get(kind, "")
            response = self._call_openai(system_prompt, user_prompt)
            if response:
                return response
            return self._render_template(template_name, context)
        # Unsupported mode
        return None

    def _render_template(self, filename: str, context: Dict[str, Any]) -> str:
        template = _load_template(filename)
        safe_context = {key: _safe(value) for key, value in context.items()}
        return template.format_map(safe_context)

    def _call_openai(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        if not self.client or HumanMessage is None or SystemMessage is None:
            return None
        try:
            messages = []
            if system_prompt:
                messages.append(SystemMessage(content=system_prompt))
            messages.append(HumanMessage(content=user_prompt))
            result = self.client.invoke(messages)
            content = getattr(result, "content", None)
            if isinstance(content, str):
                return content
            if isinstance(content, list):  # langchain can return list of message chunks
                joined = "".join(chunk.get("text", "") for chunk in content if isinstance(chunk, dict))
                return joined or None
            return str(content) if content else None
        except Exception as exc:  # pragma: no cover - network/runtime errors
            self.logger.warning("LLM call failed: %s", exc)
            return None

    def _prepare_case_note_context(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        metadata = payload.get("metadata") or {}
        plan = metadata.get("protection_plan") or {}
        scenario = metadata.get("protection_scenario") or {}
        details = payload.get("details") or {}
        context: Dict[str, Any] = {
            "timestamp": payload.get("timestamp") or datetime.now(timezone.utc).isoformat(),
            "placa": payload.get("placa") or "-",
            "market": payload.get("plaza") or payload.get("market") or "-",
            "action": payload.get("action") or "-",
            "outcome": payload.get("outcome") or "-",
            "risk_band": payload.get("risk_band") or "-",
            "reason": payload.get("reason") or "-",
            "channel": metadata.get("channel") or details.get("channel") or "-",
            "template": payload.get("template") or "-",
            "scenario_type": scenario.get("type") or payload.get("scenario") or "sin_escenario",
            "scenario_summary": self._summarize_scenario(scenario),
            "plan_type": plan.get("plan_type") or payload.get("plan_type") or "sin_plan",
            "plan_status": plan.get("status") or "sin_estado",
            "plan_valid_until": plan.get("valid_until") or "sin_fecha",
            "plan_reset_cycle_days": plan.get("reset_cycle_days") or 0,
            "protections_used": plan.get("protections_used") if plan.get("protections_used") is not None else payload.get("protections_used"),
            "protections_allowed": plan.get("protections_allowed") if plan.get("protections_allowed") is not None else payload.get("protections_allowed"),
            "notes": payload.get("notes") or "-",
            "prompt_version": "case_note_v1",
        }
        context["critical_signals"], recommended = self._derive_case_flags(plan, scenario, payload)
        context["recommended_action"] = recommended
        context["financial_snapshot"] = self._build_financial_snapshot(details, scenario)
        return context

    def _summarize_scenario(self, scenario: Dict[str, Any]) -> str:
        if not scenario:
            return "sin detalle"
        summary = scenario.get("type", "sin_tipo")
        payment_change = scenario.get("payment_change")
        if payment_change is not None:
            summary += f", Δpago {float(payment_change):+.0f}"
        term_change = scenario.get("term_change")
        if term_change is not None:
            summary += f", Δplazo {int(term_change):+d}"
        annual_irr = scenario.get("annual_irr")
        if annual_irr is not None:
            summary += f", TIR {float(annual_irr) * 100:.1f}%"
        params = scenario.get("params") or {}
        if params:
            formatted = ", ".join(f"{key}={value}" for key, value in params.items())
            summary += f" ({formatted})"
        return summary

    def _derive_case_flags(self, plan: Dict[str, Any], scenario: Dict[str, Any], payload: Dict[str, Any]) -> tuple[str, str]:
        flags = []
        actions = []
        if plan.get("status") and str(plan.get("status")).lower() == "expired":
            flags.append("plan expirado")
            actions.append("Renovar contrato antes de ejecutar protección")
        if scenario.get("requires_manual_review"):
            flags.append("requiere revisión manual")
            actions.append("Escalar a analista de riesgo")
        if payload.get("has_delinquency_flag") or (payload.get("details") or {}).get("arrears_amount", 0) > 0:
            flags.append("mora activa")
            if "Recordar pago" not in actions:
                actions.append("Coordinar promesa de pago con el operador")
        if not flags:
            flags.append("sin banderas críticas")
        recommended = "; ".join(actions) if actions else "Confirmar la opción con el operador y registrar autorización."
        return ", ".join(flags), recommended

    def _build_financial_snapshot(self, details: Dict[str, Any], scenario: Dict[str, Any]) -> str:
        lines = []
        if "expected_payment" in details:
            lines.append(f"- Pago esperado: {_format_currency(details['expected_payment'])}")
        if "arrears_amount" in details:
            lines.append(f"- Saldo vencido: {_format_currency(details['arrears_amount'])}")
        if "collected_amount" in details:
            lines.append(f"- Recaudado: {_format_currency(details['collected_amount'])}")
        if "gnv_credit_30d" in details:
            lines.append(f"- Consumo GNV 30d: {_format_currency(details['gnv_credit_30d'])}")
        if scenario.get("new_payment") is not None:
            lines.append(f"- Pago nuevo: {_format_currency(scenario['new_payment'])}")
        if scenario.get("capitalized_interest") is not None:
            lines.append(f"- Interés capitalizado: {_format_currency(scenario['capitalized_interest'])}")
        if scenario.get("annual_irr") is not None:
            lines.append(f"- TIR proyectada: {_format_percent(scenario['annual_irr'])}")
        return "\n".join(lines) if lines else "Sin datos financieros relevantes."

    def _prepare_alert_context(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        flags = payload.get("flags") or {}
        flag_parts = []
        for name, active in flags.items():
            if active:
                flag_parts.append(name.replace("_", " "))
        flag_summary = ", ".join(flag_parts) if flag_parts else "sin banderas"
        context = {
            "prompt_version": "alert_v1",
            "reference_ts": payload.get("reference_ts") or datetime.now(timezone.utc).isoformat(),
            "placa": payload.get("placa") or "-",
            "market": payload.get("market") or "-",
            "plan_type": payload.get("plan_type") or "sin_plan",
            "plan_status": payload.get("plan_status") or "sin_estado",
            "plan_valid_until": payload.get("plan_valid_until") or "sin_fecha",
            "protections_remaining": payload.get("protections_remaining") if payload.get("protections_remaining") is not None else "N/D",
            "flag_summary": flag_summary,
            "last_outcome_desc": payload.get("last_outcome_desc") or "Sin outcome reciente",
            "metric_snapshot": payload.get("metric_snapshot") or "Sin métricas adicionales",
            "impact_projection": payload.get("impact_projection") or "Revisar impacto en la próxima corrida de TIR",
            "recommended_action": payload.get("recommended_action") or "Contactar al operador y ejecutar el escenario sugerido.",
        }
        return context

    def _prepare_summary_context(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        scenarios = payload.get("scenarios") or []
        if scenarios:
            primary = scenarios[0]
            primary_desc = self._summarize_scenario(primary)
        else:
            primary_desc = "Sin escenarios viables"
        overview = "; ".join(self._summarize_scenario(item) for item in scenarios[1:]) if len(scenarios) > 1 else "-"
        context = {
            "prompt_version": "summary_v1",
            "placa": payload.get("placa") or "-",
            "market": payload.get("market") or "-",
            "balance": _format_currency(payload.get("balance")),
            "payment": _format_currency(payload.get("payment")),
            "irr_target": _format_percent(payload.get("irr_target")) if payload.get("irr_target") is not None else "-",
            "scenarios_list": "\n".join(f"- {self._summarize_scenario(item)}" for item in scenarios) if scenarios else "-",
            "critical_signals": payload.get("critical_signals") or "Sin banderas",
            "primary_scenario": primary_desc,
            "scenario_overview": overview,
            "recommended_action": payload.get("recommended_action") or "Confirma con el operador la opción que prefiera antes de ejecutar.",
        }
        return context

    def _append_jsonl(self, path: Path, data: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(data, ensure_ascii=False) + "\n")

    def _heuristic_behaviour(
        self,
        transcript: str,
        *,
        max_tags: int = 5,
    ) -> Dict[str, Any]:
        lower = transcript.lower()
        tags: list[str] = []
        notes: list[str] = []

        def add(tag: str, note: str) -> None:
            if tag not in tags:
                tags.append(tag)
                notes.append(note)

        if any(word in lower for word in ["promesa", "prometo", "me comprometo", "pago mañana"]):
            add("promise_payment", "El operador menciona una promesa de pago")
        if "no estoy usando gnv" in lower or "no uso gnv" in lower or "no he usado gnv" in lower:
            add("low_consumption", "Declara no usar GNV recientemente")
        if "no puedo pagar" in lower or "no voy a pagar" in lower or "no tengo dinero" in lower:
            add("payment_refusal", "Manifiesta imposibilidad de pago")
        if "motor" in lower and ("se apaga" in lower or "se apag" in lower or "se detiene" in lower):
            add("engine_shutdown", "Reporta apagones de motor")
        if "codigo" in lower and "p0" in lower:
            add("fault_code_reported", "Menciona código de falla")
        if "queja" in lower or "molest" in lower or "inconform" in lower:
            add("customer_complaint", "Expresa queja o inconformidad")
        if "no tengo evidencia" in lower or "sin fotos" in lower or "no pude mandar" in lower:
            add("missing_evidence", "No cuenta con evidencias solicitadas")
        if "creo que" in lower or "no recuerdo" in lower:
            add("uncertain_information", "Lenguaje incierto o evasivo")

        summary = "; ".join(notes) if notes else "sin señales relevantes"
        confidence = "high" if len(tags) >= 3 else "medium" if tags else "low"
        return {
            "summary": summary,
            "tags": tags[:max_tags],
            "behavioural_notes": notes,
            "confidence": confidence,
        }

    def _parse_behaviour_response(self, response: str) -> Optional[Dict[str, Any]]:
        text = response.strip()
        if not text:
            return None
        if text.startswith("```"):
            text = text.strip("`")
            # Remove language hint if present
            if "\n" in text:
                text = "\n".join(text.splitlines()[1:])
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return None

        summary = str(data.get("summary", "sin señales relevantes")).strip() or "sin señales relevantes"
        tags_raw = data.get("tags") or []
        if isinstance(tags_raw, str):
            tags = [tags_raw]
        else:
            tags = [str(t) for t in tags_raw if isinstance(t, (str, int, float))]
        notes_raw = data.get("behavioural_notes") or []
        if isinstance(notes_raw, str):
            notes = [notes_raw]
        else:
            notes = [str(n) for n in notes_raw if isinstance(n, (str, int, float))]
        confidence = str(data.get("confidence", "medium")).lower().strip() or "medium"
        return {
            "summary": summary,
            "tags": tags,
            "behavioural_notes": notes,
            "confidence": confidence,
        }

    def _limit_behaviour(
        self,
        data: Dict[str, Any],
        *,
        max_tags: int = 5,
    ) -> Dict[str, Any]:
        tags = [
            str(tag).strip()
            for tag in data.get("tags", [])
            if str(tag).strip()
        ]
        limited = tags[: max(1, max_tags)]
        notes = [str(n).strip() for n in data.get("behavioural_notes", []) if str(n).strip()]
        return {
            "summary": str(data.get("summary", "sin señales relevantes")).strip()
            or "sin señales relevantes",
            "tags": limited,
            "behavioural_notes": notes,
            "confidence": str(data.get("confidence", "medium")).strip() or "medium",
        }


def get_llm_service() -> Optional[LLMService]:
    global _LLM_SERVICE
    if _LLM_SERVICE is not None:
        return _LLM_SERVICE
    config = LLMConfig.from_env()
    if config.mode == "disabled":
        LOGGER.debug("PIA LLM service disabled via PIA_LLM_MODE")
        return None
    try:
        _LLM_SERVICE = LLMService(config)
    except Exception as exc:  # pragma: no cover - guard for missing deps
        LOGGER.warning("Failed to initialise LLM service: %s", exc)
        _LLM_SERVICE = None
    return _LLM_SERVICE


__all__ = ["get_llm_service", "feature_enabled", "LLMConfig", "LLMService"]
