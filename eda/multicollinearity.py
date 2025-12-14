import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns
import matplotlib.pyplot as plt

#--- Function : check_multicollinearity ---
def check_multicollinearity(df, threshold=5.0, show_heatmap=True, show_vif=True):
    """
    Analyze multicollinearity in a dataset using correlation matrix and VIF.
    Designed for regression workflows to detect unstable or redundant predictors.

    Parameters
    ----------
    df : DataFrame
        Dataset containing numerical features.
    threshold : float, optional
        VIF threshold above which a feature is considered problematic.
        Common values: 5.0 (moderate), 10.0 (severe).
    show_heatmap : bool, optional
        Whether to display a heatmap of the correlation matrix.
    show_vif : bool, optional
        Whether to display the sorted VIF table.

    Returns
    -------
    dict
        {
            "correlation_matrix": DataFrame,
            "vif": DataFrame with VIF scores,
            "high_vif_features": list of feature names,
            "has_multicollinearity": bool
        }
    """

    #Keep only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < df.shape[1]:
        raise ValueError("Input DataFrame must contain only numerical variables.")

    #Correlation matrix
    corr_matrix = numeric_df.corr(method='spearman')

    if show_heatmap:
        plt.figure(figsize=(12,10))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True, cbar=True)
        plt.title("Spearman Correlation Matrix")
        plt.show()

    #VIF calculation
    vif_data = pd.DataFrame()
    vif_data["feature"] = numeric_df.columns
    vif_data["vif"] = [variance_inflation_factor(numeric_df.values, i) 
                       for i in range(numeric_df.shape[1])]
    vif_data.sort_values("vif", ascending=False, inplace=True)

    high_vif_features = vif_data[vif_data['vif'] > threshold]['feature'].tolist()
    has_multicollinearity = len(high_vif_features) > 0

    if show_vif:
        print("Variance Inflation Factor (VIF) scores:")
        display(vif_data)

        if has_multicollinearity:
            print(f"Features with VIF > {threshold}: {high_vif_features}")
        else:
            print("No features exceed the VIF threshold. Multicollinearity is low.")

    #Return dictionary
    return {
        "correlation_matrix": corr_matrix "\n",
        "vif": vif_data "\n",
        "high_vif_features": high_vif_features "\n",
        "has_multicollinearity": has_multicollinearity
    }
