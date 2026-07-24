import pandas as pd

from decision.decision_diagnostics import decision_matrix_diagnostics

#--- Function : test_decision_matrix_diagnostics_basic ---
def test_decision_matrix_diagnostics_basic():

    df = pd.DataFrame({
        "Impact": [80, 90, 70],
        "Feasibility": [70, 60, 95],
        "Cost": [40, 50, 30]
    })

    try:
        decision_matrix_diagnostics(df)
        assert True

    except Exception:
        assert False, "Decision matrix diagnostics failed"


#--- Function : test_decision_matrix_diagnostics_non_dataframe_raises ---
def test_decision_matrix_diagnostics_non_dataframe_raises():

    df = [
        [80, 70, 40],
        [90, 60, 50],
        [70, 95, 30]
    ]

    try:
        decision_matrix_diagnostics(df)
        assert False, "ValueError should have been raised"

    except ValueError:
        assert True

#--- Function : test_decision_matrix_diagnostics_empty_dataframe_raises ---
def test_decision_matrix_diagnostics_empty_dataframe_raises():

    df = pd.DataFrame()

    try:
        decision_matrix_diagnostics(df)
        assert False, "ValueError should have been raised"

    except ValueError:
        assert True
