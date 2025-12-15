import pandas as pd
from scipy.stats import mannwhitneyu
import statsmodels.api as sm
import statsmodels.formula.api as smf

#--- Function mann_whitney_cliff ---
def mann_whitney_cliff(
    df,
    value_col,
    group_col,
    group1,
    group2,
    alternative="two-sided"
):
    """
    Perform a Mann–Whitney U test with Cliff's Delta effect size.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    value_col : str
        Name of the continuous variable to test.
    group_col : str
        Name of the grouping variable.
    group1, group2 :
        Values in group_col defining the two groups.
    alternative : {'two-sided', 'less', 'greater'}, default 'two-sided'
        Alternative hypothesis.

    Returns
    -------
    pd.Series
        Test statistic, p-value, Cliff's Delta, and sample sizes.
    """

    x = df.loc[df[group_col] == group1, value_col].dropna()
    y = df.loc[df[group_col] == group2, value_col].dropna()

    #Mann–Whitney U test
    u_stat, p_value = mannwhitneyu(x, y, alternative=alternative)

    #Cliff's Delta
    nx, ny = len(x), len(y)
    gt = sum(xi > yi for xi in x for yi in y)
    lt = sum(xi < yi for xi in x for yi in y)
    delta = (gt - lt) / (nx * ny)

    results = pd.Series({
        "Group 1": group1,
        "Group 2": group2,
        "n_group1": nx,
        "n_group2": ny,
        "U_statistic": u_stat,
        "p_value": p_value,
        "Cliffs_delta": delta
    })

    return results

#--- Function : robust_ancova ---
def robust_ancova(df, outcome, factor, covariates, estimator=sm.robust.norms.HuberT):
    """
    Perform a robust ANCOVA-like regression using RLM (Robust Linear Model).
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset containing outcome, factor, and covariates.
    outcome : str
        Name of the dependent variable (numeric).
    factor : str
        Name of the categorical independent variable (factor).
    covariates : list of str
        List of covariate column names (numeric).
    estimator : function, optional
        Robust estimator function from statsmodels (default: HuberT).
    
    Returns:
    --------
    fitted_model : RLMResults
        Fitted robust linear model object from statsmodels.
    """
    #Ensure factor is categorical
    df[factor] = df[factor].astype('category')
    
    #Build formula string
    formula = f"{outcome} ~ {factor}"
    if covariates:
        formula += " + " + " + ".join(covariates)
    
    #Fit robust linear model
    model = smf.rlm(formula=formula, data=df, M=estimator()).fit()
    
    #Print summary
    print(model.summary())
    
    return model
