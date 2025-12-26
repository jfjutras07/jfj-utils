import pandas as pd
import numpy as np
from statsmodels.stats.anova import AnovaRM
from eda.stats_param import repeated_anova

#--- Function : test_repeated_anova_basic ---
def test_repeated_anova_basic():
    df = pd.DataFrame({
        'participant': [1,1,1,2,2,2,3,3,3],
        'time': ['morning','afternoon','evening']*3,
        'stress': [5,6,4,7,6,5,6,5,5]
    })
    anova_table = repeated_anova(df, subject='participant', within='time', dv='stress')
    assert hasattr(anova_table, "anova_table") or hasattr(anova_table, "summary")  # statsmodels RMANOVA object

#--- Function : test_repeated_anova_non_numeric_raises ---
def test_repeated_anova_non_numeric_raises():
    df = pd.DataFrame({
        'participant': [1,1,1,2,2,2],
        'time': ['morning','afternoon','evening']*2,
        'stress': ['a','b','c','d','e','f']
    })
    try:
        repeated_anova(df, subject='participant', within='time', dv='stress')
        assert False, "ValueError should have been raised"
    except ValueError:
        assert True
