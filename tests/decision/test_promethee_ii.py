import pandas as pd
from decision.mcda import promethee_ii

#--- Function : test_promethee_ii_basic ---
def test_promethee_ii_basic():

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

        result = promethee_ii(
            df,
            weights,
            benefit_criteria
        )

        assert "PROMETHEE_Net_Flow" in result.columns

    except Exception:

        assert False, "PROMETHEE II calculation failed"

#--- Function : test_promethee_ii_missing_column_raises ---
def test_promethee_ii_missing_column_raises():

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

        promethee_ii(
            df,
            weights,
            ["Quality"]
        )

        assert False, "ValueError should have been raised"

    except ValueError:

        assert True
