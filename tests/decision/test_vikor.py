import pandas as pd
from decision.mcda import vikor

#--- Function : test_vikor_basic ---
def test_vikor_basic():

    df = pd.DataFrame(
        {
            "Cost": [100, 120, 90],
            "Quality": [80, 90, 70]
        },
        index=["A", "B", "C"]
    )

    weights = {
        "Cost": 0.4,
        "Quality": 0.6
    }

    benefit_criteria = [
        "Quality"
    ]

    try:

        result = vikor(
            df,
            weights,
            benefit_criteria
        )

        assert "VIKOR_S" in result.columns
        assert "VIKOR_R" in result.columns
        assert "VIKOR_Q" in result.columns

    except Exception:

        assert False, "VIKOR calculation failed"

#--- Function : test_vikor_missing_column_raises ---
def test_vikor_missing_column_raises():

    df = pd.DataFrame(
        {
            "Cost": [100, 120]
        }
    )

    weights = {
        "Cost": 0.5,
        "Quality": 0.5
    }

    try:

        vikor(
            df,
            weights,
            ["Quality"]
        )

        assert False, "ValueError should have been raised"

    except ValueError:

        assert True
