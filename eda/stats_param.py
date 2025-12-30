import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.multivariate.manova import MANOVA
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multicomp import pairwise_tukeyhsd

#---Function: ancova_test---
def ancova_test(df, dv, factor, covariates):
    """
    Perform an ANCOVA to compare means between groups while adjusting for covariates.

    Example:
    --------
    # Compare test scores between two schools adjusting for hours of study
    data = pd.DataFrame({
        'score': [85, 90, 88, 92, 78, 80, 82, 75],
        'school': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'],
        'hours_study': [5, 6, 5, 7, 4, 5, 4, 3]
    })
    ancova_result = ancova_test(data, dv='score', factor='school', covariates=['hours_study'])

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset containing dependent variable, factor, and covariates.
    dv : str
        Dependent variable (numeric).
    factor : str
        Categorical independent variable (factor).
    covariates : list of str
        List of numeric covariate column names.

    Returns:
    --------
    anova_table : pandas.DataFrame
        ANCOVA summary table with F-statistics and p-values.
    """
    #Check dependent variable
    if not pd.api.types.is_numeric_dtype(df[dv]):
        raise ValueError(f"Dependent variable {dv} must be numeric.")
    
    #Ensure factor is categorical
    df[factor] = df[factor].astype('category')
    
    #Ensure covariates are numeric
    for cov in covariates:
        if not pd.api.types.is_numeric_dtype(df[cov]):
            raise ValueError(f"Covariate {cov} must be numeric.")
    
    #Build formula
    formula = f"{dv} ~ C({factor})"
    if covariates:
        formula += " + " + " + ".join(covariates)
    
    #Fit model
    model = ols(formula, data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    
    #Print results
    print(f"ANCOVA for {dv} by {factor} adjusting for {', '.join(covariates)}")
    print(anova_table)
    
    return anova_table

#---Function: anova_test---
def anova_test(df, column, group):
    """
    Perform a one-way ANOVA to compare the means of three or more groups.

    Example:
    --------
    # Compare students' test scores across three classes
    data = pd.DataFrame({
        'score': [85, 90, 88, 92, 78, 80, 82, 75, 70],
        'class': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C']
    })
    anova_result = anova_test(data, column='score', group='class')

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset containing the numeric column and grouping variable.
    column : str
        Name of the numeric column to test.
    group : str
        Name of the categorical grouping column.

    Returns:
    --------
    anova_table : pandas.DataFrame
        ANOVA summary table containing F-statistic and p-value.
    """
    #Check column is numeric
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column {column} must be numeric.")
    
    #Ensure group is categorical
    df[group] = df[group].astype('category')
    
    #Build and fit model
    model = ols(f'{column} ~ C({group})', data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    
    #Print results
    print(f"One-way ANOVA for {column} by {group}")
    print(anova_table)
    
    return anova_table

#---Function: f_test_variance---
def f_test_variance(df, column, group, group1, group2):
    """
    Perform an F-test to compare the variances of two groups.

    Example:
    --------
    # Compare variance of blood pressure between two treatment groups
    data = pd.DataFrame({
        'bp': [120, 125, 130, 115, 128, 132, 118, 122],
        'treatment': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B']
    })
    f_stat, p_value = f_test_variance(data, column='bp', group='treatment', group1='A', group2='B')

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
    #Check column is numeric
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column {column} must be numeric.")
    
    #Extract groups and drop NaN
    data1 = df[df[group] == group1][column].dropna()
    data2 = df[df[group] == group2][column].dropna()
    
    #Compute variances
    var1 = data1.var(ddof=1)
    var2 = data2.var(ddof=1)
    f_stat = var1 / var2
    
    #Compute two-tailed p-value
    dfn = len(data1) - 1
    dfd = len(data2) - 1
    if f_stat > 1:
        p_value = 2 * (1 - stats.f.cdf(f_stat, dfn, dfd))
    else:
        p_value = 2 * stats.f.cdf(f_stat, dfn, dfd)
    
    #Print results
    print(f"F-test for variances of {column} between groups '{group1}' and '{group2}'")
    print(f"F-statistic = {f_stat:.4f}, p-value = {p_value:.4f}")
    
    return f_stat, p_value

#---Function: mancova_test---
def mancova_test(df, dependent_vars, factor, covariates):
    """
    Perform a MANCOVA to compare multiple dependent variables across groups while adjusting for covariates.

    Example:
    --------
    # Compare math and science scores across schools adjusting for hours of study
    data = pd.DataFrame({
        'math': [85, 90, 88, 92, 78, 80, 82, 75],
        'science': [80, 85, 87, 90, 70, 75, 78, 72],
        'school': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'],
        'hours_study': [5, 6, 5, 7, 4, 5, 4, 3]
    })
    result = mancova_test(data, dependent_vars=['math', 'science'], factor='school', covariates=['hours_study'])

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset containing dependent variables, factor, and covariates.
    dependent_vars : list of str
        List of numeric dependent variable column names.
    factor : str
        Categorical independent variable (factor).
    covariates : list of str
        List of numeric covariate column names.

    Returns:
    --------
    mancova_result : MANOVA
        Fitted MANCOVA object with summary results.
    """
    #Check dependent variables and covariates are numeric
    for col in dependent_vars + covariates:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"{col} must be numeric.")
    
    #Ensure factor is categorical
    df[factor] = df[factor].astype('category')
    
    #Build formula
    dv_formula = ' + '.join(dependent_vars)
    cov_formula = ' + '.join(covariates)
    formula = f"{dv_formula} ~ C({factor})"
    if covariates:
        formula += " + " + cov_formula
    
    #Fit MANCOVA
    mancova_result = MANOVA.from_formula(formula, data=df)
    
    #Print summary
    print(f"MANCOVA for {', '.join(dependent_vars)} by {factor} adjusting for {', '.join(covariates)}")
    print(mancova_result.mv_test())
    
    return mancova_result

#---Function: manova_test---
def manova_test(df, dependent_vars, factors):
    """
    Perform a MANOVA to compare multiple dependent variables across one or more categorical factors.

    Example:
    --------
    # Compare BasePay across JobTitle and Gender
    data = pd.DataFrame({
        'BasePay': [85000, 90000, 88000, 92000, 78000, 80000, 82000, 75000],
        'JobTitle': ['A', 'A', 'B', 'B', 'C', 'C', 'A', 'B'],
        'Gender': ['M', 'F', 'M', 'F', 'M', 'F', 'F', 'M']
    })
    manova_result = manova_test(data, dependent_vars=['BasePay'], factors=['JobTitle', 'Gender'])

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset containing dependent variables and categorical factors.
    dependent_vars : list of str
        List of numeric dependent variable column names.
    factors : list of str
        List of categorical independent variables (factors).

    Returns:
    --------
    manova_result : MANOVA
        Fitted MANOVA object with summary results.
    """
    #Check dependent variables are numeric
    for dv in dependent_vars:
        if not pd.api.types.is_numeric_dtype(df[dv]):
            raise ValueError(f"Dependent variable {dv} must be numeric.")
    
    #Ensure factors are categorical
    for f in factors:
        df[f] = df[f].astype('category')
    
    #Build formula
    dv_formula = ' + '.join(dependent_vars)
    factor_formula = ' + '.join([f'C({f})' for f in factors])
    formula = f"{dv_formula} ~ {factor_formula}"
    
    #Fit MANOVA
    manova_result = MANOVA.from_formula(formula, data=df)
    
    #Print summary
    print(f"MANOVA for {', '.join(dependent_vars)} by {', '.join(factors)}")
    print(manova_result.mv_test())
    
    return manova_result

#---Function: multi_factor_anova---
def multi_factor_anova(df, dv, factors):
    """
    Perform a multi-factor ANOVA to compare means across multiple categorical factors.

    Example:
    --------
    #Compare BasePay across JobTitle and Gender
    data = pd.DataFrame({
        'BasePay': [85000, 90000, 88000, 92000, 78000, 80000, 82000, 75000],
        'JobTitle': ['A', 'A', 'B', 'B', 'C', 'C', 'A', 'B'],
        'Gender': ['M', 'F', 'M', 'F', 'M', 'F', 'F', 'M']
    })
    multi_factor_anova(data, dv='BasePay', factors=['JobTitle', 'Gender'])

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset containing the dependent variable and categorical factors.
    dv : str
        Name of the numeric dependent variable.
    factors : list of str
        List of categorical factor column names.

    Returns:
    --------
    anova_table : pandas.DataFrame
        ANOVA summary table containing F-statistics and p-values.
    """
    #Ensure DV is numeric
    if not pd.api.types.is_numeric_dtype(df[dv]):
        raise ValueError(f"Dependent variable {dv} must be numeric.")
    
    #Ensure factors are categorical
    for f in factors:
        df[f] = df[f].astype('category')
    
    #Build formula
    factor_formula = ' + '.join([f'C({f})' for f in factors])
    formula = f'{dv} ~ {factor_formula}'
    
    #Fit model
    model = ols(formula, data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    
    #Print results
    print(f"Multi-factor ANOVA for {dv} by {', '.join(factors)}")
    print(anova_table)
    
    return anova_table

#---Function: one_sample_ttest---
def one_sample_ttest(df, column, popmean):
    """
    Perform a one-sample t-test.

    Example:
    --------
    # Compare the average height of plants to the expected mean of 50 cm
    data = pd.DataFrame({'height': [48, 52, 50, 49, 51]})
    t_stat, p_value = one_sample_ttest(data, column='height', popmean=50)

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
    #Check column is numeric
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column {column} must be numeric.")
    
    #Perform one-sample t-test
    t_stat, p_value = stats.ttest_1samp(df[column].dropna(), popmean)
    
    #Print results
    print(f"One-sample t-test for {column} against mean={popmean}")
    print(f"t-statistic = {t_stat:.4f}, p-value = {p_value:.4f}")
    
    return t_stat, p_value

#---Function: paired_ttest---
def paired_ttest(df, column_before, column_after):
    """
    Perform a paired (dependent) t-test between two related measurements.

    Example:
    --------
    # Compare patients' weight before and after a diet
    data = pd.DataFrame({
        'weight_before': [80, 75, 90, 85, 78],
        'weight_after': [78, 74, 88, 83, 77]
    })
    t_stat, p_value = paired_ttest(data, column_before='weight_before', column_after='weight_after')

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
    #Check columns are numeric
    if not pd.api.types.is_numeric_dtype(df[column_before]):
        raise ValueError(f"Column {column_before} must be numeric.")
    if not pd.api.types.is_numeric_dtype(df[column_after]):
        raise ValueError(f"Column {column_after} must be numeric.")
    
    #Drop NaNs
    data_before = df[column_before].dropna()
    data_after = df[column_after].dropna()
    
    #Check same length
    if len(data_before) != len(data_after):
        raise ValueError("Columns must have the same number of observations after dropping NaNs.")
    
    #Perform paired t-test
    t_stat, p_value = stats.ttest_rel(data_before, data_after)
    
    #Print results
    print(f"Paired t-test between '{column_before}' and '{column_after}'")
    print(f"t-statistic = {t_stat:.4f}, p-value = {p_value:.4f}")
    
    return t_stat, p_value

#---Function: repeated_anova---
def repeated_anova(df, subject, within, dv):
    """
    Perform a repeated measures ANOVA on a dataset with repeated measurements.

    Example:
    --------
    # Compare stress levels of participants at three times: morning, afternoon, evening
    data = pd.DataFrame({
        'participant': [1, 1, 1, 2, 2, 2, 3, 3, 3],
        'time': ['morning', 'afternoon', 'evening'] * 3,
        'stress': [5, 6, 4, 7, 6, 5, 6, 5, 5]
    })
    anova_table = repeated_anova(data, subject='participant', within='time', dv='stress')

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset containing the repeated measures data.
    subject : str
        Column name identifying the subjects.
    within : str
        Column name for the within-subject factor (repeated measure).
    dv : str
        Column name for the dependent variable (numeric).

    Returns:
    --------
    anova_table : pandas.DataFrame
        Repeated measures ANOVA summary table.
    """
    #Check DV is numeric
    if not pd.api.types.is_numeric_dtype(df[dv]):
        raise ValueError(f"Dependent variable {dv} must be numeric.")
    
    #Ensure subject and within factors are categorical
    df[subject] = df[subject].astype('category')
    df[within] = df[within].astype('category')
    
    #Fit repeated measures ANOVA
    aovrm = AnovaRM(df, depvar=dv, subject=subject, within=[within])
    anova_table = aovrm.fit()
    
    #Print results
    print(f"Repeated measures ANOVA for {dv} by {within} within subjects {subject}")
    print(anova_table)
    
    return anova_table

#---Function: tukey_posthoc---
def tukey_posthoc(df, column, group, alpha=0.05):
    """
    Perform Tukey's HSD post-hoc test for pairwise comparisons after ANOVA.

    Example:
    --------
    # Compare students' test scores across three classes
    data = pd.DataFrame({
        'score': [85, 90, 88, 92, 78, 80, 82, 75, 70],
        'class': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C']
    })
    tukey_result = tukey_posthoc(data, column='score', group='class')

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
    #Check column is numeric
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column {column} must be numeric.")
    
    #Ensure group is categorical
    df[group] = df[group].astype('category')
    
    #Perform Tukey HSD
    tukey_result = pairwise_tukeyhsd(endog=df[column], groups=df[group], alpha=alpha)
    
    #Print summary
    print(f"Tukey HSD post-hoc test for {column} by {group}")
    print(tukey_result.summary())
    
    return tukey_result

#---Function: two_sample_ttest---
def two_sample_ttest(df, column, group, group1, group2):
    """
    Perform an independent two-sample t-test between two groups.

    Example:
    --------
    # Compare the average blood pressure between two treatment groups
    data = pd.DataFrame({
        'bp': [120, 125, 130, 115, 128, 132, 118, 122],
        'treatment': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B']
    })
    t_stat, p_value = two_sample_ttest(data, column='bp', group='treatment', group1='A', group2='B')

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
    #Check column is numeric
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column {column} must be numeric.")
    
    #Extract the two groups and drop NaNs
    data1 = df[df[group] == group1][column].dropna()
    data2 = df[df[group] == group2][column].dropna()
    
    #Perform independent two-sample t-test
    t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=True)
    
    #Print results
    print(f"Two-sample t-test for {column} between groups '{group1}' and '{group2}'")
    print(f"t-statistic = {t_stat:.4f}, p-value = {p_value:.4f}")
    
    return t_stat, p_value

#---Function: welch_anova_test---
def welch_anova_test(df, column, group):
    """
    Perform a one-way Welch ANOVA to compare the means of three or more groups,
    robust to unequal variances.

    Example:
    --------
    # Compare students' test scores across three classes
    data = pd.DataFrame({
        'score': [85, 90, 88, 92, 78, 80, 82, 75, 70],
        'class': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C']
    })
    welch_result = welch_anova_test(data, column='score', group='class')

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset containing the numeric column and grouping variable.
    column : str
        Name of the numeric column to test.
    group : str
        Name of the categorical grouping column.

    Returns:
    --------
    welch_table : pandas.DataFrame
        Welch ANOVA summary table.
    """
    #Check column is numeric
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column {column} must be numeric.")
    
    #Ensure group is categorical
    df[group] = df[group].astype('category')
    
    #Build OLS model
    model = ols(f'{column} ~ C({group})', data=df).fit()
    
    #Perform Welch ANOVA using HC3 robust covariance
    welch_table = sm.stats.anova_lm(model, typ=2, robust='hc3')
    
    #Print results
    print(f"Welch ANOVA for {column} by {group}")
    print(welch_table)
    
    return welch_table


