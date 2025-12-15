import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.stats import mannwhitneyu

#--- Function : numeric_skew_kurt ---
def numeric_skew_kurt(df, numeric_cols):
    """
    Compute skewness and kurtosis for selected numeric columns.

    Parameters:
    - df: pandas DataFrame
    - numeric_cols: list of numeric columns to analyze (must be explicitly provided)

    Returns:
    - skew_kurt_df: DataFrame with skewness and kurtosis for each column
    """
    results = []
    
    for col in numeric_cols:
        if col not in df.columns:
            print(f"Warning: {col} not found in DataFrame. Skipping.")
            continue
            
        col_data = df[col].dropna()
        results.append({
            "Column": col,
            "Skewness": skew(col_data),
            "Kurtosis": kurtosis(col_data)
        })
    
    skew_kurt_df = pd.DataFrame(results)
    return skew_kurt_df

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

