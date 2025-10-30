import os
import tempfile
from datetime import datetime, timezone

from agents.pia.src.cobranza import (
    CobranzaCase,
    determinar_estrategia_cobranza,
    preparar_cobranza_payload,
)


def test_determinar_estrategia_tramos() -> None:
    case = CobranzaCase(placa="ABC", dias_mora=2)
    strategy = determinar_estrategia_cobranza(case)
    assert strategy.tone == "suave"

    case_mid = CobranzaCase(placa="ABC", dias_mora=6)
    strategy_mid = determinar_estrategia_cobranza(case_mid)
    assert strategy_mid.include_protection_offer is True

    case_late = CobranzaCase(placa="ABC", dias_mora=9)
    strategy_late = determinar_estrategia_cobranza(case_late)
    assert strategy_late.prepare_advisor_context is True


def test_preparar_cobranza_payload_generates_message_without_escalation(monkeypatch) -> None:
    case = CobranzaCase(
        placa="ABC",
        dias_mora=2,
        monto_vencido=1500,
        client_context={"nombre": "Luis"},
    )

    # Fijar elección aleatoria
    monkeypatch.setattr("random.choice", lambda seq: seq[0])

    payload = preparar_cobranza_payload(case)
    assert payload["strategy"]["tone"] == "suave"
    assert "Luis" in payload["message"]
    assert payload["advisor_alert"] is False


def test_preparar_cobranza_payload_escalates_and_writes_file(monkeypatch) -> None:
    case = CobranzaCase(
        placa="XYZ",
        dias_mora=9,
        monto_vencido=3400,
        intentos_contacto=4,
        ultima_respuesta=datetime(2025, 1, 1, tzinfo=timezone.utc),
    )
    monkeypatch.setattr("random.choice", lambda seq: seq[0])
    with tempfile.TemporaryDirectory() as tmpdir:
        alerts_path = os.path.join(tmpdir, "alerts.jsonl")
        monkeypatch.setenv("PIA_COBRANZA_ALERTS_PATH", alerts_path)
        payload = preparar_cobranza_payload(case)
        assert payload["advisor_alert"] is True
        with open(alerts_path, "r", encoding="utf-8") as handle:
            lines = handle.readlines()
        assert len(lines) == 1
        assert "\"placa\": \"XYZ\"" in lines[0]
