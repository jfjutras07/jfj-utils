import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

# Importing centralized style constants
from .style import SEQUENTIAL_CMAP, DEFAULT_FIGSIZE, GREY_DARK

#---Function: chi_square_heatmap---
def chi_square_heatmap(df, col1, col2, figsize=DEFAULT_FIGSIZE, cmap=SEQUENTIAL_CMAP):
    """
    Generate a heatmap showing the contribution of each cell to the Chi-square statistic
    between two categorical variables. Only the visual representation is displayed.
    """

    # Contingency table (direct crosstab is more performant than manual casting)
    contingency = pd.crosstab(df[col1], df[col2])
    
    # Chi-square test to get expected frequencies
    chi2, p, dof, expected = chi2_contingency(contingency)
    
    # Contribution to chi-square: (Observed - Expected)^2 / Expected
    contrib = (contingency - expected) ** 2 / expected
    contrib_percent = (contrib / contrib.sum().sum()) * 100
    
    # Plot heatmap using explicit figure and axes for stability
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        contrib_percent,
        annot=True,
        fmt=".1f",
        cmap=cmap,
        cbar_kws={'label': 'Contribution (%)'},
        ax=ax,
        annot_kws={"color": GREY_DARK}  # Consistent with BI text style
    )
    
    # Text and labels
    ax.set_title(f"Chi-square contributions: {col1} vs {col2}", fontweight='bold', pad=15)
    ax.set_ylabel(col1)
    ax.set_xlabel(col2)
    
    plt.tight_layout()
    plt.show()
