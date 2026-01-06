import pandas as pd
import numpy as np

def robust_data_check(
    df, 
    numeric_threshold=0.9,  # seuil corrélation forte
    iqr_factor=1.5,         # seuil outliers IQR
    correlation_method='pearson',
    verbose=True
):
    """
    Perform a comprehensive dataset check before modeling.
    Returns only anomalies or issues; prints "All checks passed" if none.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to check.
    numeric_threshold : float
        Correlation coefficient above which columns are flagged.
    iqr_factor : float
        Multiplier for IQR to detect outliers.
    correlation_method : str
        'pearson', 'spearman', etc. method for correlation.
    verbose : bool
        If True, prints details.
    """
    
    issues_found = False
    report = []

    # ---- 1. Missing or infinite values ----
    missing = df.isna().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        issues_found = True
        report.append("Missing values detected:")
        report.append(missing)
    
    inf_vals = df.isin([np.inf, -np.inf]).sum()
    inf_vals = inf_vals[inf_vals > 0]
    if not inf_vals.empty:
        issues_found = True
        report.append("Infinite values detected:")
        report.append(inf_vals)
    
    # ---- 2. Duplicate columns ----
    duplicated_cols = df.columns[df.columns.duplicated()].tolist()
    if duplicated_cols:
        issues_found = True
        report.append(f"Duplicated columns detected: {duplicated_cols}")
    
    # ---- 3. Outliers detection for numeric columns ----
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        outliers = {}
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - iqr_factor * IQR
            upper = Q3 + iqr_factor * IQR
            count_out = ((df[col] < lower) | (df[col] > upper)).sum()
            if count_out > 0:
                outliers[col] = count_out
        if outliers:
            issues_found = True
            report.append("Outliers detected (IQR method):")
            report.append(outliers)
    
    # ---- 4. Correlations ----
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr(method=correlation_method)
        high_corr = np.where((abs(corr_matrix) > numeric_threshold) & (abs(corr_matrix) < 1.0))
        high_corr_pairs = [(numeric_cols[i], numeric_cols[j], corr_matrix.iloc[i,j])
                           for i,j in zip(*high_corr) if i < j]
        if high_corr_pairs:
            issues_found = True
            report.append(f"Highly correlated pairs (>|{numeric_threshold}|):")
            report.append(high_corr_pairs)
    
    # ---- 5. Basic type summary ----
    if verbose:
        type_summary = df.dtypes.value_counts()
        report.append("Column type summary:")
        report.append(type_summary)
    
    # ---- 6. Optional: check for expected PCA columns ----
    pca_cols = [c for c in df.columns if '_PC' in c]
    if pca_cols:
        report.append(f"PCA columns detected: {len(pca_cols)} columns ({pca_cols[:5]}...)")
    
    # ---- Print report ----
    if issues_found:
        for r in report:
            print(r)
    else:
        print("All checks passed. Dataset looks good for modeling.")
