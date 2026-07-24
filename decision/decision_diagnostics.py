import pandas as pd
import numpy as np

#--- Function : decision_matrix_diagnostics ---
def decision_matrix_diagnostics(df):
    """
    Diagnose a decision matrix before multi-criteria decision analysis.

    Parameters
    ----------
    df : pandas.DataFrame
        Decision matrix where rows represent alternatives
        and columns represent criteria.
    """

    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")

    if df.empty:
        raise ValueError("Decision matrix cannot be empty")

    print("--- Decision Matrix Diagnostics ---")

    print(f"Number of alternatives : {df.shape[0]}")
    print(f"Number of criteria     : {df.shape[1]}")

    print("\nData Types:")
    print(df.dtypes)

    missing_values = df.isnull().sum().sum()

    print("\nMissing Values:")
    print(f"Total missing values : {missing_values}")

    print("\nCriteria Statistics:")
    print(
        df.describe().T[
            ["mean", "std", "min", "max"]
        ].to_string()
    )

    print("\nCriteria Variability:")

    variability = pd.DataFrame({
        "Mean": df.mean(),
        "Std": df.std(),
        "Coefficient_of_Variation": (
            df.std() / df.mean().replace(0, np.nan)
        )
    })

    print(variability.to_string())

    print("\nScale Range:")

    scale_range = pd.DataFrame({
        "Minimum": df.min(),
        "Maximum": df.max(),
        "Range": df.max() - df.min()
    })

    print(scale_range.to_string())

    print("-" * 40)
