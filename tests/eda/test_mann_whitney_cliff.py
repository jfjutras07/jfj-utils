import pandas as pd
import numpy as np
from eda.stats_non_param import mann_whitney_cliff

#--- Function : test_mann_whitney_cliff_basic ---
def test_mann_whitney_cliff_basic():
    df = pd.DataFrame({
        "score": [1, 2, 3, 4, 5, 6, 7, 8],
        "group": ["A", "A", "A", "A", "B", "B", "B", "B"]
    })
    res = mann_whitney_cliff(df, "score", "group", "A", "B")

    assert isinstance(res, pd.Series)
    assert all(key in res.index for key in ["Group 1", "Group 2", "n_group1", "n_group2", "U_statistic", "p_value", "Cliffs_delta"])
    assert res["n_group1"] == 4
    assert res["n_group2"] == 4

#--- Function : test_mann_whitney_cliff_effect_direction ---
def test_mann_whitney_cliff_effect_direction():
    df = pd.DataFrame({
        "score": [1, 1, 1, 1, 10, 10, 10, 10],
        "group": ["A", "A", "A", "A", "B", "B", "B", "B"]
    })
    res = mann_whitney_cliff(df, "score", "group", "A", "B")

    assert res["Cliffs_delta"] < 0  # group1 < group2

#--- Function : test_mann_whitney_cliff_two_sided_pvalue ---
def test_mann_whitney_cliff_two_sided_pvalue():
    df = pd.DataFrame({
        "score": np.random.randint(1, 10, 20),
        "group": ["A"]*10 + ["B"]*10
    })
    res = mann_whitney_cliff(df, "score", "group", "A", "B", alternative="two-sided")

    assert 0 <= res["p_value"] <= 1

#--- Function : test_mann_whitney_cliff_less_greater_alternative ---
def test_mann_whitney_cliff_less_greater_alternative():
    df = pd.DataFrame({
        "score": [1, 2, 3, 4, 5, 6],
        "group": ["A", "A", "A", "B", "B", "B"]
    })
    res_less = mann_whitney_cliff(df, "score", "group", "A", "B", alternative="less")
    res_greater = mann_whitney_cliff(df, "score", "group", "A", "B", alternative="greater")

    assert res_less["p_value"] != res_greater["p_value"]
