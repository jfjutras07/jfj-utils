import pandas as pd
import numpy as np
import scikit_posthocs as sp
from eda.stats_non_param import dunn_posthoc

#--- Function : test_dunn_posthoc_basic ---
def test_dunn_posthoc_basic():
    df = pd.DataFrame({
        "score": [10,12,14,20,22,24,30,32,34],
        "group": ["A"]*3 + ["B"]*3 + ["C"]*3
    })
    result = dunn_posthoc(df, "score", "group")
    assert isinstance(result, pd.DataFrame)
    assert all(g in result.columns for g in ["A","B","C"])
    assert all(g in result.index for g in ["A","B","C"])

#--- Function : test_dunn_posthoc_non_numeric_raises ---
def test_dunn_posthoc_non_numeric_raises():
    df = pd.DataFrame({
        "score": ["a","b","c","d","e","f","g","h","i"],
        "group": ["A"]*3 + ["B"]*3 + ["C"]*3
    })
    try:
        dunn_posthoc(df, "score", "group")
        assert False, "ValueError should have been raised"
    except ValueError:
        assert True
