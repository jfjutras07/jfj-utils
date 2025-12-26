import pandas as pd
from scipy import stats
from scipy.stats import mannwhitneyu
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scikit_posthocs as sp
import numpy as np
from skbio.stats.distance import DistanceMatrix
from skbio.stats.distance import permanova
from scipy.spatial.distance import pdist, squareform

#--- Function: dunn_friedman_posthoc ---
def dunn_friedman_posthoc(df, dv, subject, within, p_adjust='bonferroni'):
    """
    Perform Dunn's post-hoc test for pairwise comparisons after a Friedman test.

    Example:
    --------
    # Compare stress levels of participants at three times: morning, afternoon, evening
    data = pd.DataFrame({
        'participant': [1, 1, 1, 2, 2, 2, 3, 3, 3],
        'time': ['morning', 'afternoon', 'evening'] * 3,
        'stress': [5, 6, 4, 7, 6, 5, 6, 5, 5]
    })
    posthoc_result = dunn_friedman_posthoc(data, dv='stress', subject='participant', within='time')

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset containing the repeated measures data.
    dv : str
        Column name for the dependent variable (numeric).
    subject : str
        Column name identifying the subjects.
    within : str
        Column name for the within-subject factor (repeated measure).
    p_adjust : str, optional
        Method for p-value adjustment ('bonferroni', 'holm', etc.), default='bonferroni'.

    Returns:
    --------
    posthoc_result : pandas.DataFrame
        Pairwise p-values adjusted according to the chosen method.
    """
    #Ensure DV is numeric
    if not pd.api.types.is_numeric_dtype(df[dv]):
        raise ValueError(f"Dependent variable {dv} must be numeric.")
    
    #Ensure subject and within factor are categorical
    df[subject] = df[subject].astype('category')
    df[within] = df[within].astype('category')
    
    #Pivot data: rows = subjects, columns = within levels
    pivot_df = df.pivot(index=subject, columns=within, values=dv)
    
    #Check for NaNs
    if pivot_df.isnull().any().any():
        raise ValueError("Missing values detected. Each subject must have all repeated measures.")
    
    #Perform Dunn post-hoc
    posthoc_result = sp.posthoc_dunn(pivot_df, p_adjust=p_adjust)
    
    #Print results
    print(f"Dunn post-hoc for {dv} by {within} within subjects {subject} (p-adjust: {p_adjust})")
    print(posthoc_result)
    
    return posthoc_result

#--- Function: dunn_posthoc ---
def dunn_posthoc(df, column, group, p_adjust='bonferroni'):
    """
    Perform Dunn's post-hoc test for pairwise comparisons after a Kruskal-Wallis test.

    Example:
    --------
    # Compare students' test scores across three classes
    data = pd.DataFrame({
        'score': [85, 90, 88, 92, 78, 80, 82, 75, 70],
        'class': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C']
    })
    dunn_result = dunn_posthoc(data, column='score', group='class')

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset containing the numeric column and grouping variable.
    column : str
        Name of the numeric column to test.
    group : str
        Name of the categorical grouping column.
    p_adjust : str, optional
        Method for p-value adjustment ('bonferroni', 'holm', 'fdr_bh', etc.), default='bonferroni'.

    Returns:
    --------
    dunn_result : pandas.DataFrame
        Pairwise p-values adjusted according to the chosen method.
    """
    #Ensure column is numeric
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column {column} must be numeric.")
    
    #Ensure group is categorical
    df[group] = df[group].astype('category')
    
    #Perform Dunn post-hoc test
    dunn_result = sp.posthoc_dunn(df, val_col=column, group_col=group, p_adjust=p_adjust)
    
    #Print results
    print(f"Dunn post-hoc test for {column} by {group} (p-adjust method: {p_adjust})")
    print(dunn_result)
    
    return dunn_result

#--- Function: friedman_test ---
def friedman_test(df, dv, subject, within):
    """
    Perform a Friedman test for repeated measures on a dataset.

    Example:
    --------
    # Compare stress levels of participants at three times: morning, afternoon, evening
    data = pd.DataFrame({
        'participant': [1, 1, 1, 2, 2, 2, 3, 3, 3],
        'time': ['morning', 'afternoon', 'evening'] * 3,
        'stress': [5, 6, 4, 7, 6, 5, 6, 5, 5]
    })
    stat, p_value = friedman_test(data, dv='stress', subject='participant', within='time')

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset containing the repeated measures data.
    dv : str
        Column name for the dependent variable (numeric).
    subject : str
        Column name identifying the subjects.
    within : str
        Column name for the within-subject factor (repeated measure).

    Returns:
    --------
    stat : float
        Computed Friedman statistic.
    p_value : float
        Two-tailed p-value for the test.
    """
    #Ensure DV is numeric
    if not pd.api.types.is_numeric_dtype(df[dv]):
        raise ValueError(f"Dependent variable {dv} must be numeric.")
    
    #Ensure subject and within factor are categorical
    df[subject] = df[subject].astype('category')
    df[within] = df[within].astype('category')
    
    #Pivot data: rows = subjects, columns = within levels
    pivot_df = df.pivot(index=subject, columns=within, values=dv)
    
    #Check for NaNs
    if pivot_df.isnull().any().any():
        raise ValueError("Missing values detected. Each subject must have all repeated measures.")
    
    #Perform Friedman test
    stat, p_value = stats.friedmanchisquare(*[pivot_df[col] for col in pivot_df.columns])
    
    #Print results
    print(f"Friedman test for {dv} by {within} within subjects {subject}")
    print(f"Friedman statistic = {stat:.4f}, p-value = {p_value:.4f}")
    
    return stat, p_value

#--- Function: kruskal_wallis_test ---
def kruskal_wallis_test(df, column, group):
    """
    Perform a Kruskal-Wallis H test to compare the distributions of three or more groups.

    Example:
    --------
    # Compare students' test scores across three classes
    data = pd.DataFrame({
        'score': [85, 90, 88, 92, 78, 80, 82, 75, 70],
        'class': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C']
    })
    stat, p_value = kruskal_wallis_test(data, column='score', group='class')

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
    stat : float
        Computed Kruskal-Wallis H statistic.
    p_value : float
        Two-tailed p-value for the test.
    """
    #Ensure column is numeric
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column {column} must be numeric.")
    
    #Ensure group is categorical
    df[group] = df[group].astype('category')
    
    #Extract data for each group
    groups_data = [df[df[group] == lvl][column].dropna() for lvl in df[group].cat.categories]
    
    #Perform Kruskal-Wallis test
    stat, p_value = stats.kruskal(*groups_data)
    
    #Print results
    print(f"Kruskal-Wallis test for {column} by {group}")
    print(f"H-statistic = {stat:.4f}, p-value = {p_value:.4f}")
    
    return stat, p_value

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

#--- Function: permanova_test ---
def permanova_test(df, features, group_col, method='euclidean', permutations=999):
    """
    Perform a PERMANOVA test to compare group differences on multivariate data.

    Example:
    --------
    # Compare multivariate species abundances across sites
    data = pd.DataFrame({
        'species1': [5, 3, 6, 7, 2, 4],
        'species2': [7, 8, 5, 6, 4, 3],
        'site': ['A', 'A', 'B', 'B', 'C', 'C']
    })
    result = permanova_test(data, features=['species1', 'species2'], group_col='site')

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset containing multivariate features and grouping variable.
    features : list of str
        Names of columns with multivariate features.
    group_col : str
        Name of the categorical grouping column.
    method : str, optional
        Distance metric (default='euclidean').
    permutations : int, optional
        Number of permutations for significance testing (default=999).

    Returns:
    --------
    result : skbio.stats.distance._base.PERMANOVAResults
        PERMANOVA result object containing F-statistic and p-value.
    """
    #Ensure features are numeric
    for f in features:
        if not pd.api.types.is_numeric_dtype(df[f]):
            raise ValueError(f"Feature {f} must be numeric.")
    
    #Ensure group column is categorical
    df[group_col] = df[group_col].astype('category')
    
    #Compute distance matrix
    dist_matrix = squareform(pdist(df[features], metric=method))
    dm = DistanceMatrix(dist_matrix, ids=df.index.astype(str))
    
    #Run PERMANOVA
    result = permanova(dm, df[group_col], permutations=permutations)
    
    #Print results
    print(f"PERMANOVA for features {features} by {group_col} using {method} distance")
    print(f"F-statistic = {result['test statistic']:.4f}, p-value = {result['p-value']:.4f}")
    
    return result

#--- Placeholder: Quade test (R) ---
def quade_test_placeholder():
    """
    Quade test (non-parametric ANCOVA-like test).

    Note:
    -----
    This test is available in R via the 'quade.test' function.
    To run it in R, you can use:

        quade.test(response ~ factor + covariate, data=mydata)

    Parameters:
    -----------
    None in Python. Execution is done in R.

    Returns:
    --------
    None. Prints a message explaining how to use the test in R.
    """
    print("Quade test is available in R via quade.test().")
    print("Example R usage:")
    print("  quade.test(response ~ factor + covariate, data=mydata)")


#--- Placeholder: Aligned Rank Transform (ART) test (R) ---
def art_test_placeholder():
    """
    ART test (Aligned Rank Transform for non-parametric factorial ANOVA).

    Note:
    -----
    This test is available in R via the 'ART' package.
    Typical workflow in R:

        library(ART)
        model <- art(response ~ factor1 * factor2 + covariate, data=mydata)
        anova(model)

    Parameters:
    -----------
    None in Python. Execution is done in R.

    Returns:
    --------
    None. Prints a message explaining how to use the test in R.
    """
    print("ART test is available in R via the ART package.")
    print("Example R usage:")
    print("  library(ART)")
    print("  model <- art(response ~ factor1 * factor2 + covariate, data=mydata)")
    print("  anova(model)")

