import pandas as pd
from scipy import stats
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

#--- Function: one_sample_wilcoxon ---
def one_sample_wilcoxon(df, column, popmedian):
    """
    Perform a one-sample Wilcoxon signed-rank test.

    Example:
    --------
    # Compare the median height of plants to the expected median of 50 cm
    data = pd.DataFrame({'height': [48, 52, 50, 49, 51]})
    stat, p_value = one_sample_wilcoxon(data, column='height', popmedian=50)

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset containing the numeric column to test.
    column : str
        Name of the numeric column to test.
    popmedian : float
        The population median to compare against.

    Returns:
    --------
    stat : float
        Computed Wilcoxon test statistic.
    p_value : float
        Two-tailed p-value for the test.
    """
    #Ensure column is numeric
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column {column} must be numeric.")

    #Perform Wilcoxon signed-rank test
    stat, p_value = stats.wilcoxon(df[column].dropna() - popmedian)
    
    #Print results
    print(f"One-sample Wilcoxon test for {column} against median={popmedian}")
    print(f"statistic = {stat:.4f}, p-value = {p_value:.4f}")
    
    return stat, p_value

#--- Function: paired_wilcoxon ---
def paired_wilcoxon(df, column_before, column_after):
    """
    Perform a paired Wilcoxon signed-rank test between two related measurements.

    Example:
    --------
    # Compare patients' weight before and after a diet
    data = pd.DataFrame({
        'weight_before': [80, 75, 90, 85, 78],
        'weight_after': [78, 74, 88, 83, 77]
    })
    stat, p_value = paired_wilcoxon(data, column_before='weight_before', column_after='weight_after')

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
    stat : float
        Computed Wilcoxon test statistic.
    p_value : float
        Two-tailed p-value for the test.
    """
    #Ensure columns are numeric
    if not pd.api.types.is_numeric_dtype(df[column_before]):
        raise ValueError(f"Column {column_before} must be numeric.")
    if not pd.api.types.is_numeric_dtype(df[column_after]):
        raise ValueError(f"Column {column_after} must be numeric.")
    
    #Drop NaN values and ensure same length
    data_before = df[column_before].dropna()
    data_after = df[column_after].dropna()
    if len(data_before) != len(data_after):
        raise ValueError("Columns must have the same number of observations after dropping NaNs.")

    #Perform Wilcoxon signed-rank test
    stat, p_value = stats.wilcoxon(data_before, data_after)
    
    #Print results
    print(f"Paired Wilcoxon test between '{column_before}' and '{column_after}'")
    print(f"statistic = {stat:.4f}, p-value = {p_value:.4f}")
    
    return stat, p_value

