import numpy as np
import pandas as pd

from agents.hase.src.synthetic import SyntheticScenario, generate_synthetic_rows


def _dummy_base(n: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    return pd.DataFrame(
        {
            "plaza_limpia": rng.choice(["EDOMEX - ECATEPEC", "AGUASCALIENTES"], size=n),
            "placa": [f"TEST-{i:04d}" for i in range(n)],
            "coverage_ratio_30d": rng.uniform(0.2, 1.5, size=n),
            "coverage_ratio_14d": rng.uniform(0.2, 1.5, size=n),
            "coverage_ratio_7d": rng.uniform(0.2, 1.5, size=n),
            "downtime_days_14d": rng.uniform(0, 5, size=n),
            "downtime_days_30d": rng.uniform(0, 10, size=n),
            "downtime_days_7d": rng.uniform(0, 3, size=n),
            "litros_30d": rng.uniform(100, 500, size=n),
            "recaudo_30d": rng.uniform(1000, 5000, size=n),
            "default_flag": rng.integers(0, 2, size=n),
            "label_reason": "real",
        }
    )


def test_generate_synthetic_rows_shapes():
    base = _dummy_base()
    scenarios = {
        "consumption_gap": SyntheticScenario("consumption_gap", 20, 1, "synthetic_consumption_gap"),
        "stable_high": SyntheticScenario("stable_high", 15, 0, "synthetic_stable_high"),
    }
    synthetic = generate_synthetic_rows(base, scenarios, rng=np.random.default_rng(42))
    assert len(synthetic) == 35
    assert set(["consumption_gap", "stable_high"]).issubset(set(scenarios.keys()))
    assert synthetic["default_flag"].sum() >= 20
    assert synthetic["label_reason"].str.contains("synthetic").all()
    assert (synthetic.select_dtypes(include=["number"]) >= 0).all().all()
