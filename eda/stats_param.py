import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.multivariate.manova import MANOVA
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multicomp import pairwise_tukeyhsd

#---Function: ancova_test---
def ancova_test(df, dv, factor, covariates, return_model=False):
    """
    Perform an ANCOVA to compare means between groups while adjusting for covariates.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset containing the variables.
    dv : str
        Name of the dependent variable (numeric).
    factor : str
        Name of the categorical factor.
    covariates : list of str
        Names of numeric covariates.
    return_model : bool, optional
        If True, return the fitted model along with ANOVA table.

    Returns:
    --------
    anova_table : pd.DataFrame
        ANOVA table.
    model : statsmodels.regression.linear_model.RegressionResultsWrapper
        Fitted OLS model.
    """
    if not pd.api.types.is_numeric_dtype(df[dv]):
        raise ValueError(f"Dependent variable {dv} must be numeric.")
    df[factor] = df[factor].astype('category')
    for cov in covariates:
        if not pd.api.types.is_numeric_dtype(df[cov]):
            raise ValueError(f"Covariate {cov} must be numeric.")
    
    # Build formula with main effects and factor x covariate interactions
    main_effects = f"C({factor})"
    covariate_terms = " + ".join(covariates)
    interaction_terms = " + ".join([f"C({factor}):{cov}" for cov in covariates])

    formula = f"{dv} ~ {main_effects}"
    if covariates:
        formula += " + " + covariate_terms + " + " + interaction_terms

    model = ols(formula, data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    
    if return_model:
        return anova_table, model
    return anova_table, model

#---Function: anova_test---
def anova_test(df, column, group, return_model=False):
    """
    Perform a one-way ANOVA to compare the means of three or more groups.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset containing the numeric column and grouping variable.
    column : str
        Name of the numeric column.
    group : str
        Name of the categorical grouping variable.
    return_model : bool, optional
        If True, return the fitted model along with ANOVA table.

    Returns:
    --------
    anova_table : pd.DataFrame
        ANOVA table.
    model : statsmodels.regression.linear_model.RegressionResultsWrapper
        Fitted OLS model.
    """
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column {column} must be numeric.")
    df[group] = df[group].astype('category')
    model = ols(f'{column} ~ C({group})', data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    return anova_table, model

#---Function: f_test_variance---
def f_test_variance(df, column, group, group1, group2):
    """
    Perform an F-test to compare the variances of two groups.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset containing the numeric column and grouping variable.
    column : str
        Name of the numeric column.
    group : str
        Name of the categorical grouping column.
    group1 : str
        Name of the first group to compare.
    group2 : str
        Name of the second group to compare.

    Returns:
    --------
    f_stat : float
        Computed F-statistic (variance ratio).
    p_value : float
        Two-tailed p-value for the F-test.
    """
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column {column} must be numeric.")
    
    data1 = df[df[group] == group1][column].dropna()
    data2 = df[df[group] == group2][column].dropna()
    
    var1 = data1.var(ddof=1)
    var2 = data2.var(ddof=1)
    f_stat = var1 / var2
    
    dfn = len(data1) - 1
    dfd = len(data2) - 1
    if f_stat > 1:
        p_value = 2 * (1 - stats.f.cdf(f_stat, dfn, dfd))
    else:
        p_value = 2 * stats.f.cdf(f_stat, dfn, dfd)
    
    print(f"F-test for variances of {column} between groups '{group1}' and '{group2}'")
    print(f"F-statistic = {f_stat:.4f}, p-value = {p_value:.4f}")
    
    return f_stat, p_value

#---Function: mancova_test---
def mancova_test(df, dependent_vars, factor, covariates, return_model=True):
    """
    Perform a MANCOVA to compare multiple dependent variables across groups while adjusting for covariates.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset containing the dependent variables, factor, and covariates.
    dependent_vars : list of str
        Names of numeric dependent variables.
    factor : str
        Name of the categorical factor.
    covariates : list of str
        Names of numeric covariates.
    return_model : bool, optional
        If True, return the MANCOVA model.

    Returns:
    --------
    model : MANOVA
        Fitted MANCOVA model.
    """
    for col in dependent_vars + covariates:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"{col} must be numeric.")
    df[factor] = df[factor].astype('category')

    dv_formula = ' + '.join(dependent_vars)
    cov_formula = ' + '.join(covariates)

    # Build formula with main effects and factor x covariate interactions
    interaction_terms = ' + '.join([f'C({factor}):{cov}' for cov in covariates])

    formula = f"{dv_formula} ~ C({factor})"
    if covariates:
        formula += " + " + cov_formula + " + " + interaction_terms

    model = MANOVA.from_formula(formula, data=df)
    print(f"MANCOVA for {', '.join(dependent_vars)} by {factor} adjusting for {', '.join(covariates)}")
    print(model.mv_test())

    return model

#---Function: manova_test---
def manova_test(df, dependent_vars, factors, return_model=True):
    """
    Perform a MANOVA to compare multiple dependent variables across categorical factors.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset containing the dependent variables and factors.
    dependent_vars : list of str
        Names of numeric dependent variables.
    factors : list of str
        Names of categorical factors.
    return_model : bool, optional
        If True, return the MANOVA model.

    Returns:
    --------
    model : MANOVA
        Fitted MANOVA model.
    """
    for col in dependent_vars:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"{col} must be numeric.")
    for f in factors:
        df[f] = df[f].astype('category')

    dv_formula = ' + '.join(dependent_vars)

    # Build formula with main effects and interactions between factors
    factor_formula = ' * '.join([f'C({f})' for f in factors])

    formula = f"{dv_formula} ~ {factor_formula}"

    model = MANOVA.from_formula(formula, data=df)
    print(f"MANOVA for {', '.join(dependent_vars)} by {', '.join(factors)}")
    print(model.mv_test())

    return model

#---Function: multi_factor_anova---
def multi_factor_anova(df, dv, factors, return_model=False):
    """
    Perform a multi-factor ANOVA to compare means across multiple categorical factors.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset containing the dependent variable and factors.
    dv : str
        Name of the numeric dependent variable.
    factors : list of str
        Names of categorical factors.
    return_model : bool, optional
        If True, return the fitted model along with ANOVA table.

    Returns:
    --------
    anova_table : pd.DataFrame
        ANOVA table.
    model : statsmodels.regression.linear_model.RegressionResultsWrapper
        Fitted OLS model.
    """
    for f in factors:
        df[f] = df[f].astype('category')

    # Build formula with main effects and interactions between factors
    formula = f'{dv} ~ ' + ' * '.join([f'C({f})' for f in factors])

    model = ols(formula, data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    return anova_table, model

#---Function: one_sample_ttest---
def one_sample_ttest(df, column, popmean):
    """
    Perform a one-sample t-test.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset containing the numeric column to test.
    column : str
        Name of the numeric column to test.
    popmean : float
        The population mean to compare against.

    Returns:
    --------
    t_stat : float
        Computed t-statistic.
    p_value : float
        Two-tailed p-value for the test.
    """
    t_stat, p_value = stats.ttest_1samp(df[column].dropna(), popmean)
    print(f"One-sample t-test for {column} against mean={popmean}")
    print(f"t-statistic = {t_stat:.4f}, p-value = {p_value:.4f}")
    return t_stat, p_value

#---Function: paired_ttest---
def paired_ttest(df, column_before, column_after):
    """
    Perform a paired (dependent) t-test between two related measurements.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset containing the two numeric columns to test.
    column_before : str
        Name of the first numeric column (e.g., before intervention).
    column_after : str
        Name of the second numeric column (e.g., after intervention).

    Returns:
    --------
    t_stat : float
        Computed t-statistic.
    p_value : float
        Two-tailed p-value for the test.
    """
    df_pair = df[[column_before, column_after]].dropna()
    t_stat, p_value = stats.ttest_rel(df_pair[column_before], df_pair[column_after])
    print(f"Paired t-test between '{column_before}' and '{column_after}'")
    print(f"t-statistic = {t_stat:.4f}, p-value = {p_value:.4f}")
    return t_stat, p_value

#---Function: repeated_anova---
def repeated_anova(df, subject, within, dv, return_model=False):
    """
    Perform a repeated measures ANOVA on a dataset with repeated measurements.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset containing the dependent variable and within-subject factor.
    subject : str
        Column representing subject IDs.
    within : str
        Column representing the within-subject factor.
    dv : str
        Name of the numeric dependent variable.
    return_model : bool, optional
        If True, return the fitted model along with ANOVA result.

    Returns:
    --------
    result : AnovaRMResults
        Fitted repeated measures ANOVA result.
    aovrm : AnovaRM
        Original AnovaRM model.
    """
    df[subject] = df[subject].astype('category')
    df[within] = df[within].astype('category')
    aovrm = AnovaRM(df, depvar=dv, subject=subject, within=[within])
    result = aovrm.fit()
    return result, aovrm

#---Function: robust_anova---
#---Function: robust_anova---
def robust_anova(df, dv, factors, return_model=False):
    """
    Perform a robust ANOVA (Welch-like) for one or multiple categorical factors.
    Uses OLS with heteroscedasticity-robust errors (HC3).
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset containing the dependent variable and factors.
    dv : str
        Name of the numeric dependent variable.
    factors : list of str
        One or more categorical factors.
    return_model : bool, optional
        If True, return the fitted model along with ANOVA table.

    Returns:
    --------
    anova_table : pd.DataFrame
        Robust ANOVA table.
    model : statsmodels.regression.linear_model.RegressionResultsWrapper
        Fitted OLS model (if return_model=True).
    """
    # Check dependent variable
    if not pd.api.types.is_numeric_dtype(df[dv]):
        raise ValueError(f"Dependent variable {dv} must be numeric.")

    # Ensure all factors are categorical
    for f in factors:
        df[f] = df[f].astype('category')

    # Build formula with main effects and interactions
    formula = dv + ' ~ ' + ' * '.join([f'C({f})' for f in factors])

    # Fit model
    model = ols(formula, data=df).fit()

    # Robust ANOVA table (HC3)
    anova_table = sm.stats.anova_lm(model, typ=2, robust='hc3')

    if return_model:
        return anova_table, model
    return anova_table

#---Function: tukey_posthoc---
def tukey_posthoc(df, column, group, alpha=0.05):
    """
    Perform Tukey's HSD post-hoc test for pairwise comparisons after ANOVA.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset containing the numeric column and grouping variable.
    column : str
        Name of the numeric column to test.
    group : str
        Name of the categorical grouping column.
    alpha : float, optional
        Significance level (default 0.05).

    Returns:
    --------
    tukey_result : TukeyHSDResults
        Object containing pairwise comparisons with mean differences, p-values, and confidence intervals.
    """

    # Ensure numeric column
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column {column} must be numeric.")

    # Ensure group is categorical
    df[group] = df[group].astype('category')

    # Extract Series to ensure 1D arrays
    endog = df[column].values
    groups = df[group].values

    # Run Tukey HSD
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    tukey_result = pairwise_tukeyhsd(endog=endog, groups=groups, alpha=alpha)

    print(f"Tukey HSD post-hoc test for {column} by {group}")
    print(tukey_result.summary())
    return tukey_result

#---Function: two_sample_ttest---
def two_sample_ttest(df, column, group, group1, group2):
    """
    Perform an independent two-sample t-test between two groups.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset containing the numeric column and grouping variable.
    column : str
        Name of the numeric column to test.
    group : str
        Name of the categorical grouping column.
    group1 : str
        Name of the first group to compare.
    group2 : str
        Name of the second group to compare.

    Returns:
    --------
    t_stat : float
        Computed t-statistic.
    p_value : float
        Two-tailed p-value for the test.
    """
    data1 = df[df[group]==group1][column].dropna()
    data2 = df[df[group]==group2][column].dropna()
    t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=True)
    print(f"Two-sample t-test for {column} between '{group1}' and '{group2}'")
    print(f"t-statistic = {t_stat:.4f}, p-value = {p_value:.4f}")
    return t_stat, p_value

#---Function: welch_anova_test---
def welch_anova_test(df, column, group, return_model=False):
    """
    Perform a one-way Welch ANOVA to compare the means of three or more groups,
    robust to unequal variances.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset containing the numeric column and grouping variable.
    column : str
        Name of the numeric column.
    group : str
        Name of the categorical grouping variable.
    return_model : bool, optional
        If True, return the fitted model along with Welch ANOVA table.

    Returns:
    --------
    welch_table : pd.DataFrame
        Welch ANOVA table.
    model : statsmodels.regression.linear_model.RegressionResultsWrapper
        Fitted OLS model.
    """
    df[group] = df[group].astype('category')
    model = ols(f'{column} ~ C({group})', data=df).fit()
    welch_table = sm.stats.anova_lm(model, typ=2, robust='hc3')
    return welch_table, model
