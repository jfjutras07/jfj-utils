import pandas as pd
import numpy as np
from scipy import stats
from eda.stats_non_param import one_sample_wilcoxon

#--- Function : test_one_sample_wilcoxon_basic ---
def test_one_sample_wilcoxon_basic():
    df = pd.DataFrame({"value": [48, 52, 50, 49, 51]})
    stat, p_value = one_sample_wilcoxon(df, "value", popmedian=50)
    assert isinstance(stat, float)
    assert isinstance(p_value, float)

#--- Function : test_one_sample_wilcoxon_identical_to_median ---
def test_one_sample_wilcoxon_identical_to_median():
    df = pd.DataFrame({"value": [5, 5, 5, 5]})
    stat, p_value = one_sample_wilcoxon(df, "value", popmedian=5)
    # All values equal to median => stat = 0, p ≈ 1
    assert np.isclose(stat, 0.0)
    assert np.isclose(p_value, 1.0)

#--- Function : test_one_sample_wilcoxon_non_numeric_raises ---
def test_one_sample_wilcoxon_non_numeric_raises():
    df = pd.DataFrame({"value": ["a", "b", "c"]})
    try:
        one_sample_wilcoxon(df, "value", popmedian=5)
        assert False, "ValueError should have been raised"
    except ValueError:
        assert True
