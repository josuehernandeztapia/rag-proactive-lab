from agents.pia.src.contracts import get_contract_for_placa


def test_get_contract_for_placa_returns_dummy_entry():
    contract = get_contract_for_placa("demo-001")
    assert contract is not None
    assert contract.plan_type == "proteccion_total"
    assert contract.protections_allowed == 3
    assert contract.protections_used == 1


def test_get_contract_for_placa_missing_returns_none():
    assert get_contract_for_placa("unknown-999") is None


def test_get_contract_for_placa_manual_flag():
    contract = get_contract_for_placa("DEMO-005")
    assert contract is not None
    assert contract.requires_manual_review is True
