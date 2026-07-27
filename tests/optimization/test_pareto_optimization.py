import pandas as pd

from optimization.multiobjective_optimization import pareto_optimization

#--- Function : test_pareto_optimization_basic ---
def test_pareto_optimization_basic():

    df = pd.DataFrame(
        {
            "Cost": [100, 120, 150],
            "Quality": [80, 90, 70]
        },
        index=["A", "B", "C"]
    )

    objectives = {
        "Cost": "min",
        "Quality": "max"
    }

    try:

        result = pareto_optimization(
            df,
            objectives
        )

        assert len(result) == 2

    except Exception:

        assert False, "Pareto optimization failed"

#--- Function : test_pareto_optimization_missing_column_raises ---
def test_pareto_optimization_missing_column_raises():

    df = pd.DataFrame(
        {
            "Cost": [100, 120]
        }
    )

    objectives = {
        "Cost": "min",
        "Quality": "max"
    }

    try:

        pareto_optimization(
            df,
            objectives
        )

        assert False, "ValueError should have been raised"

    except ValueError:

        assert True
