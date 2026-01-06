import pandas as pd
import numpy as np

def premodeling_regression_check(df, target=None, corr_threshold=0.7):
    """
    Checks a dataset before regression modeling to detect potential issues.
    
    Parameters:
    - df: pd.DataFrame, dataset to check
    - target: str, name of target column (optional)
    - corr_threshold: float, correlation threshold to flag
    
    Returns:
    - dict with detected issues or "Dataset is ready for regression modeling."
    """
    issues = {}

    # 1️⃣ Missing values
    na_counts = df.isna().sum()
    na_cols = na_counts[na_counts > 0]
    if not na_cols.empty:
        issues['missing_values'] = na_cols.to_dict()

    # 2️⃣ Constant / quasi-constant columns
    const_cols = [col for col in df.columns if df[col].nunique() <= 1]
    if const_cols:
        issues['constant_columns'] = const_cols

    # 3️⃣ Outliers (IQR method)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    outliers = {}
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outlier_count = ((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)).sum()
        if outlier_count > 0:
            outliers[col] = int(outlier_count)
    if outliers:
        issues['outliers'] = {
            'total_outliers': sum(outliers.values()),
            'per_column': outliers
        }

    # 4️⃣ High correlations
    corr_matrix = df[numeric_cols].corr().abs()
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] >= corr_threshold:
                high_corr_pairs.append(
                    (corr_matrix.columns[i], corr_matrix.columns[j], float(corr_matrix.iloc[i, j]))
                )
    if high_corr_pairs:
        issues['high_correlations'] = high_corr_pairs

    # 5️⃣ Target checks
    if target:
        if target not in df.columns:
            issues['target_missing'] = f"Target column '{target}' not found."
        else:
            if df[target].isna().any():
                issues['target_missing_values'] = df[target].isna().sum()
            if df[target].nunique() <= 1:
                issues['target_constant'] = True
            if not pd.api.types.is_numeric_dtype(df[target]):
                issues['target_non_numeric'] = True

    # 6️⃣ Non-numeric columns (may break regression models)
    non_numeric_cols = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])]
    if non_numeric_cols:
        issues['non_numeric_columns'] = non_numeric_cols

    # 7️⃣ Dataset size warning
    if len(df) < 5 * len(numeric_cols):
        issues['small_dataset_warning'] = f"Rows ({len(df)}) < 5x number of numeric features ({len(numeric_cols)}). Risk of overfitting."

    if not issues:
        return "Dataset is ready for regression modeling."
    return issues
