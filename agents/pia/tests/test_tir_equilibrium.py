import pytest

from agents.pia.src.tir_equilibrium_engine import (
    DEFAULT_POLICY,
    ProtectionContext,
    ProtectionScenarioResult,
    ScenarioParams,
    evaluate_protection_scenarios,
    generate_scenario,
    load_policy_from_config,
    select_viable_scenarios,
)


def test_generate_scenario_respects_tir_minimum():
    context = ProtectionContext(
        market="edomex",
        balance=520_000,
        payment=19_000,
        term_months=48,
    )
    params = ScenarioParams("DEFER", months=3)
    result = generate_scenario(context, DEFAULT_POLICY, params)
    assert result is not None
    assert result.annual_irr >= DEFAULT_POLICY.tir_min_by_market["edomex"] - 1e-6
    assert result.irr_ok is True


def test_consumption_gap_limits_deferral():
    context = ProtectionContext(
        market="aguascalientes",
        balance=400_000,
        payment=15_000,
        term_months=36,
        has_consumption_gap=True,
    )
    scenarios = evaluate_protection_scenarios(context, DEFAULT_POLICY)
    # Only 3 month deferrals should appear when gap flag is active
    months = sorted(
        params.params.months for params in scenarios if params.type == "DEFER"
    )
    assert months == [3]


def test_restructures_cap_blocks_generation():
    context = ProtectionContext(
        market="aguascalientes",
        balance=300_000,
        payment=11_000,
        term_months=24,
        restructures_used=3,
        restructures_allowed=3,
    )
    scenarios = evaluate_protection_scenarios(context, DEFAULT_POLICY)
    assert scenarios == []


def test_protections_cap_blocks_generation():
    context = ProtectionContext(
        market="edomex",
        balance=450_000,
        payment=18_000,
        term_months=36,
        protections_used=2,
        protections_allowed=2,
    )
    scenarios = evaluate_protection_scenarios(context, DEFAULT_POLICY)
    assert scenarios == []


def test_select_viable_prefers_lower_payment_change():
    scenario_low = ProtectionScenarioResult(
        type="DEFER",
        params=ScenarioParams("DEFER", months=3),
        new_payment=12_000,
        new_term=30,
        cash_flows=[-300_000, 0, 0, 12_000] + [12_000] * 27,
        annual_irr=0.265,
        irr_target=0.25,
        irr_ok=True,
        payment_change=-500,
        payment_change_pct=-4.0,
        term_change=-6,
        capitalized_interest=5_000,
    )
    scenario_high = ProtectionScenarioResult(
        type="RECALENDAR",
        params=ScenarioParams("RECALENDAR", months=6),
        new_payment=12_900,
        new_term=42,
        cash_flows=[-300_000] + [0] * 6 + [12_900] * 36,
        annual_irr=0.26,
        irr_target=0.25,
        irr_ok=True,
        payment_change=900,
        payment_change_pct=7.2,
        term_change=6,
        capitalized_interest=12_000,
    )
    ordering = select_viable_scenarios([scenario_high, scenario_low])
    assert ordering[0] is scenario_low
    assert ordering[1] is scenario_high


def test_load_policy_from_config_overrides(tmp_path):
    config_path = tmp_path / "financial.yml"
    config_path.write_text(
        """
markets:
  nuevo:
    tir_min: 0.31
policy:
  tir_tolerance_bps: 40
  max_restructures: 3
  max_deferral_months: 4
  max_stepdown_months: 5
  max_stepdown_reduction: 0.4
  soft_deferral_cap_with_gap: 2
  soft_stepdown_cap_with_gap: 0.2
""".strip()
    )

    policy = load_policy_from_config(config_path)

    assert pytest.approx(policy.tir_min_by_market["nuevo"], rel=1e-3) == 0.31
    assert policy.tir_tolerance_bps == 40
    assert policy.max_restructures == 3
    assert policy.max_deferral_months == 4
    assert policy.max_stepdown_months == 5
    assert pytest.approx(policy.max_stepdown_reduction, rel=1e-3) == 0.4
    assert policy.soft_deferral_cap_with_gap == 2
    assert pytest.approx(policy.soft_stepdown_cap_with_gap, rel=1e-3) == 0.2


def test_defer_stepdown_scenario_generates_viable():
    context = ProtectionContext(
        market="edomex",
        balance=520_000,
        payment=19_000,
        term_months=48,
    )
    scenarios = evaluate_protection_scenarios(context, DEFAULT_POLICY)
    combo = [scenario for scenario in scenarios if scenario.type == "DEFER_STEPDOWN"]
    assert combo, "Expected DEFER_STEPDOWN scenario to be generated"
    scenario = combo[0]
    assert scenario.params.deferral_months > 0
    assert scenario.params.stepdown_months > 0
    assert scenario.irr_ok is True


def test_balloon_scenario_adds_balloon_payment():
    context = ProtectionContext(
        market="aguascalientes",
        balance=420_000,
        payment=16_000,
        term_months=42,
    )
    scenarios = evaluate_protection_scenarios(context, DEFAULT_POLICY)
    balloons = [scenario for scenario in scenarios if scenario.type == "BALLOON"]
    assert balloons, "Expected BALLOON scenario to be generated"
    scenario = balloons[0]
    assert scenario.new_term == context.term_months + 1
    assert scenario.cash_flows[-1] > 0
    assert scenario.irr_ok is True


def test_stepdown_deep_requires_manual_review():
    context = ProtectionContext(
        market="edomex",
        balance=500_000,
        payment=18_500,
        term_months=48,
        requires_manual_review=True,
    )
    scenarios = evaluate_protection_scenarios(context, DEFAULT_POLICY)
    deep = [scenario for scenario in scenarios if scenario.type == "STEPDOWN_DEEP"]
    assert deep, "Expected STEPDOWN_DEEP scenario"
    scenario = deep[0]
    assert scenario.requires_manual_review is True
    assert scenario.params.reduction_factor <= 0.3


def test_recalendar_partial_selects_min_extra():
    context = ProtectionContext(
        market="aguascalientes",
        balance=480_000,
        payment=15_500,
        term_months=36,
    )
    scenarios = evaluate_protection_scenarios(context, DEFAULT_POLICY)
    partials = [scenario for scenario in scenarios if scenario.type == "RECALENDAR_PARTIAL"]
    assert partials, "Expected RECALENDAR_PARTIAL scenario"
    scenario = partials[0]
    assert scenario.params.deferral_months >= 1
    assert scenario.new_term >= context.term_months + scenario.params.deferral_months
    assert scenario.irr_ok is True


def test_select_viable_prioritizes_non_manual_review():
    context = ProtectionContext(
        market="edomex",
        balance=520_000,
        payment=19_000,
        term_months=48,
    )
    scenarios = evaluate_protection_scenarios(context, DEFAULT_POLICY)
    manual_flags = {scenario.type: scenario.requires_manual_review for scenario in scenarios}
    assert manual_flags.get("STEPDOWN_DEEP", False) is True
    viable = select_viable_scenarios(scenarios)
    assert viable, "Viable scenarios expected"
    assert viable[0].requires_manual_review is False
    assert any(s.requires_manual_review for s in viable), "Expected at least one manual-review scenario"

