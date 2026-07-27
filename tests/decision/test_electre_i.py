import pandas as pd
from decision.mcda import electre_i

#--- Function : test_electre_i_basic ---
def test_electre_i_basic():

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

        result = electre_i(
            df,
            weights,
            benefit_criteria
        )

        assert result.shape == (3, 3)

    except Exception:

        assert False, "ELECTRE I calculation failed"

#--- Function : test_electre_i_missing_column_raises ---
def test_electre_i_missing_column_raises():

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

        electre_i(
            df,
            weights,
            ["Quality"]
        )

        assert False, "ValueError should have been raised"

    except ValueError:

        assert True
