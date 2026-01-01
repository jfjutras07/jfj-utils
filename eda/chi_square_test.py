import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

#--- Function: chi_square_test ---
def chi_square_test(df, col1, col2, show_table=False):
    """
    Perform a Chi-square test of independence between two categorical variables,
    and compute Cramér's V to measure the strength of association.

    Example:
    --------
    chi_square_result = chi_square_test(df, col1='Gender', col2='JobTitle')

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset containing the two categorical columns.
    col1 : str
        Name of the first categorical column.
    col2 : str
        Name of the second categorical column.
    show_table : bool, default False
        Whether to print the contingency table.

    Returns:
    --------
    result_dict : dict
        Dictionary containing:
        - 'chi2': Chi-square statistic
        - 'p_value': p-value
        - 'dof': degrees of freedom
        - 'cramers_v': Cramér's V effect size
    """
    #Ensure columns are categorical
    df[col1] = df[col1].astype('category')
    df[col2] = df[col2].astype('category')
    
    #Create contingency table
    contingency = pd.crosstab(df[col1], df[col2])
    
    #Chi-square test
    chi2, p, dof, expected = chi2_contingency(contingency)
    
    #Cramér's V
    n = contingency.sum().sum()
    cramers_v = np.sqrt(chi2 / (n * (min(contingency.shape)-1)))
    
    #Print results
    if show_table:
        print(f"Contingency Table: {col1} vs {col2}")
        print(contingency)
    
    print(f"\nChi-square test for {col1} vs {col2}:")
    print(f"Chi2 = {chi2:.2f}, p-value = {p:.4f}, dof = {dof}")
    print(f"Cramér's V = {cramers_v:.3f}")
