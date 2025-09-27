import numpy as np
import pandas as pd

from agents.pia.src.chain import PIAgent, build_pia_agent
from agents.pia.src.simulator import augment_dataframe, summarize_scenarios


def test_build_pia_agent_exposes_tools():
    agent = build_pia_agent()
    tool_names = {tool.name for tool in agent.tools}
    assert {"consult_hase", "simulate_pia", "trigger_make_webhook"}.issubset(tool_names)


def test_consult_hase_tool_uses_stub_scoring():
    agent = build_pia_agent()
    payload = {
        "placa": "LAB-002",
        "coverage_ratio_14d": 0.52,
        "coverage_ratio_30d": 0.58,
        "downtime_hours_30d": 80,
        "expected_payment": 11000,
        "bank_transfer": 3000,
        "gnv_credit_30d": 4200,
        "arrears_amount": 2800,
    }
    consult = next(tool for tool in agent.tools if tool.name == "consult_hase")
    result = consult.func(payload)

    assert result["placa"] == "LAB-002"
    assert 0.0 <= result["risk_score"] <= 1.0
    assert 0.0 <= result["probability_default"] <= 1.0


def test_pia_agent_simulate_decision_dict():
    agent: PIAgent = build_pia_agent()
    result = agent.simulate(
        {
            "placa": "TEST",
            "risk_score": 0.9,
            "coverage_ratio_14d": 0.4,
            "coverage_ratio_30d": 0.6,
            "downtime_hours_30d": 180,
            "arrears_amount": 2000,
            "expected_payment": 11000,
        }
    )
    assert result["action"] == "offer_protection"
    assert result["template"] == "PIA_OPCIONES"


def test_simulator_variations_and_summary():
    df = pd.DataFrame(
        {
            "placa": ["A1", "A2", "A3", "A4"],
            "risk_score": [0.6, 0.65, 0.7, 0.75],
            "coverage_ratio_14d": [0.8, 0.85, 0.9, 0.95],
            "coverage_ratio_30d": [0.85, 0.88, 0.9, 0.92],
            "downtime_hours_30d": [30, 20, 15, 10],
            "arrears_amount": [0, 0, 0, 0],
            "bank_transfer": [5000, 4800, 4600, 4500],
            "gnv_credit_30d": [5000, 5200, 5400, 5600],
            "expected_payment": [11000, 11000, 11000, 11000],
        }
    )
    augmented = augment_dataframe(df, np.random.default_rng(0))
    assert {"base", "late_payment", "prepayment", "consumption_gap", "fault_alert"}.issubset(
        set(augmented["scenario"])
    )
    summary = summarize_scenarios(augmented)
    assert {
        "plaza_limpia",
        "scenario",
        "intent",
        "registros",
        "distancia_media",
        "consumo_gap_ratio",
    }.issubset(
        summary.columns
    )
    assert (summary["consumo_gap_ratio"] >= 0).all()


def test_summarize_handles_missing_scenario_column():
    df = pd.DataFrame(
        {
            "placa": ["B1", "B2"],
            "risk_score": [0.5, 0.6],
            "arrears_amount": [100, 200],
            "coverage_ratio_14d": [0.9, 0.85],
            "coverage_ratio_30d": [0.92, 0.9],
            "downtime_hours_30d": [10, 20],
        }
    )
    summary = summarize_scenarios(df)
    assert "base" in set(summary["scenario"])
    assert summary["registros"].gt(0).all()
