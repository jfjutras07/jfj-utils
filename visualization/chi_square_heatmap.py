import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

#--- Function: chi_square_heatmap ---
def chi_square_heatmap(df, col1, col2, figsize=(10,6), cmap='Reds'):
    """
    Generate a heatmap showing the contribution of each cell to the Chi-square statistic
    between two categorical variables. Only the visual representation is displayed.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset containing the two categorical columns.
    col1 : str
        Name of the first categorical column.
    col2 : str
        Name of the second categorical column.
    figsize : tuple
        Size of the figure (width, height).
    cmap : str
        Colormap for the heatmap.

    Returns:
    --------
    None
    """
    # Ensure columns are categorical
    df[col1] = df[col1].astype('category')
    df[col2] = df[col2].astype('category')
    
    # Contingency table
    contingency = pd.crosstab(df[col1], df[col2])
    
    # Chi-square test to get expected frequencies
    chi2, p, dof, expected = chi2_contingency(contingency)
    
    # Contribution to chi-square
    contrib = (contingency - expected)**2 / expected
    contrib_percent = contrib / contrib.sum().sum() * 100  # percentage contribution
    
    # Plot heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(contrib_percent, annot=True, fmt=".1f", cmap=cmap, cbar_kws={'label': 'Contribution (%)'})
    plt.title(f"Chi-square contributions: {col1} vs {col2}")
    plt.ylabel(col1)
    plt.xlabel(col2)
    plt.tight_layout()
    plt.show()
