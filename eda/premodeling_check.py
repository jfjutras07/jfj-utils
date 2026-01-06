import numpy as np
import pandas as pd

def premodeling_regression_check(
    df: pd.DataFrame,
    target: str | None = None,
    corr_threshold: float = 0.7,
    outlier_iqr_factor: float = 1.5
):
    """
    Expert-level pre-modeling validation for regression tasks.
    Returns only problematic findings, or a clean readiness summary.
    """

    report = []
    issues_found = False

    # -------------------------------
    # Helper: infer statistical type
    # -------------------------------
    def infer_feature_type(series: pd.Series):
        if series.dropna().nunique() <= 2:
            return "binary"
        if (
            series.dropna().nunique() <= 10
            and np.all(np.mod(series.dropna(), 1) == 0)
        ):
            return "ordinal"
        return "continuous"

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    feature_types = {
        col: infer_feature_type(df[col])
        for col in numeric_cols
    }

    continuous_cols = [
        col for col, t in feature_types.items() if t == "continuous"
    ]

    # -------------------------------
    # Missing values
    # -------------------------------
    missing = df.isna().sum()
    missing = missing[missing > 0]

    if not missing.empty:
        issues_found = True
        report.append("# Missing values detected")
        for col, cnt in missing.items():
            report.append(f"- {col}: {cnt} missing values")
    else:
        report.append("# Missing values")
        report.append("No missing values detected.")

    # -------------------------------
    # Constant / near-constant columns
    # -------------------------------
    constant_cols = [
        col for col in numeric_cols if df[col].nunique() <= 1
    ]

    if constant_cols:
        issues_found = True
        report.append("\n# Constant features detected")
        for col in constant_cols:
            report.append(f"- {col}")
    else:
        report.append("\n# Feature variance")
        report.append("No constant columns detected.")

    # -------------------------------
    # Outliers (continuous only)
    # -------------------------------
    outlier_summary = {}
    total_outliers = 0

    for col in continuous_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1

        if iqr == 0:
            continue

        lower = q1 - outlier_iqr_factor * iqr
        upper = q3 + outlier_iqr_factor * iqr

        count = ((df[col] < lower) | (df[col] > upper)).sum()
        if count > 0:
            outlier_summary[col] = int(count)
            total_outliers += int(count)

    if outlier_summary:
        issues_found = True
        report.append("\n# Outliers detected (continuous features only)")
        report.append(f"- Total outliers: {total_outliers}")
        for col, cnt in outlier_summary.items():
            report.append(f"- {col}: {cnt}")
    else:
        report.append("\n# Outliers")
        report.append("No significant outliers detected in continuous features.")

    # -------------------------------
    # Correlation analysis
    # -------------------------------
    corr_pairs = []

    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr().abs()
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        for col in upper.columns:
            for idx in upper.index:
                val = upper.loc[idx, col]
                if pd.notna(val) and val >= corr_threshold:
                    corr_pairs.append((idx, col, round(val, 3)))

    if corr_pairs:
        issues_found = True
        report.append(
            "\n# High correlations detected (risk for linear and regularized models)"
        )
        for a, b, v in corr_pairs:
            report.append(f"- {a} ↔ {b}: {v}")
    else:
        report.append("\n# Feature correlations")
        report.append(
            f"No feature pairs exceed correlation threshold ({corr_threshold})."
        )

    # -------------------------------
    # Target validation
    # -------------------------------
    if target is not None:
        if target not in df.columns:
            issues_found = True
            report.append("\n# Target validation")
            report.append(f"- Target column '{target}' not found.")
        elif df[target].nunique() <= 1:
            issues_found = True
            report.append("\n# Target validation")
            report.append("- Target has no variance.")
        else:
            report.append("\n# Target validation")
            report.append("No issues detected with target variable.")

    # -------------------------------
    # Dataset size vs dimensionality
    # -------------------------------
    n_rows, n_features = df.shape

    if n_features > n_rows:
        issues_found = True
        report.append("\n# Dataset size risk")
        report.append(
            f"- High dimensionality: {n_features} features for {n_rows} rows"
        )
    else:
        report.append("\n# Dataset size")
        report.append("No size-related risks detected.")

    # -------------------------------
    # Final summary
    # -------------------------------
    if not issues_found:
        report.append(
            "\n# Summary\n"
            "No missing values. "
            "No constant features. "
            "No problematic outliers. "
            "No high correlations. "
            "Data is ready for regression modeling."
        )

    return "\n".join(report)
