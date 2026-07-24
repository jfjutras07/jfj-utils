import pandas as pd

from decision.mcda import topsis

#--- Function : test_topsis_basic ---
def test_topsis_basic():

    df = pd.DataFrame({
        "Impact": [80, 90, 70],
        "Feasibility": [70, 60, 95],
        "Cost": [40, 50, 30]
    })

    weights = {
        "Impact": 0.4,
        "Feasibility": 0.4,
        "Cost": 0.2
    }

    try:
        topsis(
            df,
            weights,
            benefit_criteria=["Impact", "Feasibility"]
        )

        assert True

    except Exception:
        assert False, "TOPSIS calculation failed"

#--- Function : test_topsis_missing_column_raises ---
def test_topsis_missing_column_raises():

    df = pd.DataFrame({
        "Impact": [80, 90],
        "Cost": [40, 50]
    })

    weights = {
        "Impact": 0.5,
        "Feasibility": 0.5
    }

    try:
        topsis(
            df,
            weights,
            benefit_criteria=["Impact"]
        )

        assert False, "ValueError should have been raised"

    except ValueError:
        assert True
