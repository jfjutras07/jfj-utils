import numpy as np
import pandas as pd


def premodeling_regression_check(
    df: pd.DataFrame,
    target: str | None = None,
    corr_threshold: float = 0.7,
    top_outliers: int = 5,
    top_correlations: int = 10,
    min_unique_for_continuous: int = 10,
    iqr_multiplier: float = 1.5
) -> str:
    """
    Expert-level pre-modeling validation for regression tasks.
    Returns a structured, readable diagnostic report.
    """

    report = []

    # --------------------------------------------------
    # Missing values
    # --------------------------------------------------
    missing = df.isnull().sum()
    missing = missing[missing > 0]

    report.append("# Missing values")
    if missing.empty:
        report.append("No missing values detected.")
    else:
        report.append("Missing values detected:")
        for col, cnt in missing.items():
            report.append(f"- {col}: {cnt}")

    # --------------------------------------------------
    # Numeric feature validation
    # --------------------------------------------------
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    report.append("\n# Feature types")
    if len(numeric_cols) == df.shape[1]:
        report.append("No non-numeric columns detected.")
    else:
        non_numeric = sorted(set(df.columns) - set(numeric_cols))
        report.append("Non-numeric columns detected (require encoding):")
        for col in non_numeric:
            report.append(f"- {col}")

    # --------------------------------------------------
    # Constant / quasi-constant features
    # --------------------------------------------------
    constant_cols = [c for c in numeric_cols if df[c].nunique(dropna=True) <= 1]

    report.append("\n# Feature variance")
    if not constant_cols:
        report.append("No constant columns detected.")
    else:
        report.append("Constant columns detected (should be removed):")
        for col in constant_cols:
            report.append(f"- {col}")

    # --------------------------------------------------
    # Continuous feature detection (heuristic)
    # --------------------------------------------------
    continuous_cols = [
        c for c in numeric_cols
        if df[c].nunique(dropna=True) >= min_unique_for_continuous
    ]

    # --------------------------------------------------
    # Outlier detection (continuous features only)
    # --------------------------------------------------
    outlier_counts = {}
    total_outliers = 0

    for col in continuous_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        if iqr <= 0:
            continue

        lower = q1 - iqr_multiplier * iqr
        upper = q3 + iqr_multiplier * iqr
        count = int(((df[col] < lower) | (df[col] > upper)).sum())

        if count > 0:
            outlier_counts[col] = count
            total_outliers += count

    report.append("\n# Outliers (continuous features only)")
    if not outlier_counts:
        report.append("No significant outliers detected.")
    else:
        sorted_outliers = sorted(
            outlier_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )

        report.append(f"- Total outliers detected: {total_outliers}")
        report.append("- Top contributing features:")
        for i, (col, cnt) in enumerate(sorted_outliers[:top_outliers], 1):
            report.append(f"  {i}. {col}: {cnt}")

        remaining = len(sorted_outliers) - top_outliers
        if remaining > 0:
            report.append(
                f"- {remaining} additional features with minor outlier presence"
            )

    # --------------------------------------------------
    # Correlation analysis (numeric only)
    # --------------------------------------------------
    corr_pairs = []

    if len(numeric_cols) >= 2:
        corr_matrix = df[numeric_cols].corr().abs()

        for i in range(1, len(corr_matrix.columns)):
            for j in range(i):
                val = corr_matrix.iloc[i, j]
                if val >= corr_threshold:
                    corr_pairs.append((
                        corr_matrix.columns[j],
                        corr_matrix.columns[i],
                        float(val)
                    ))

    report.append(f"\n# High correlations (|r| ≥ {corr_threshold:.2f})")
    if not corr_pairs:
        report.append("No problematic correlations detected.")
    else:
        corr_pairs = sorted(corr_pairs, key=lambda x: x[2], reverse=True)

        report.append("- Strongest correlations:")
        for i, (c1, c2, val) in enumerate(corr_pairs[:top_correlations], 1):
            report.append(f"  {i}. {c1} ↔ {c2}: {val:.3f}")

        remaining = len(corr_pairs) - top_correlations
        if remaining > 0:
            report.append(
                f"- {remaining} additional correlated pairs above threshold"
            )

    # --------------------------------------------------
    # Target validation
    # --------------------------------------------------
    report.append("\n# Target validation")
    if target is None:
        report.append("No target specified.")
    elif target not in df.columns:
        report.append("Target column not found in dataset.")
    else:
        if df[target].isnull().any():
            report.append("Target contains missing values.")
        elif df[target].nunique() <= 1:
            report.append("Target has zero variance.")
        else:
            report.append("No issues detected with target variable.")

    # --------------------------------------------------
    # Dataset size sanity check
    # --------------------------------------------------
    report.append("\n# Dataset size")
    if df.shape[0] < 30:
        report.append("Dataset may be too small for reliable regression.")
    else:
        report.append("No size-related risks detected.")

    # --------------------------------------------------
    # Final verdict
    # --------------------------------------------------
    report.append("\n# Final assessment")
    if (
        missing.empty
        and not constant_cols
        and not corr_pairs
        and not outlier_counts
    ):
        report.append(
            "No structural issues detected. "
            "Data is ready for regression modeling."
        )
    else:
        report.append(
            "Dataset is usable for regression, but issues above should be reviewed "
            "depending on model choice (linear, regularized, tree-based, or neural)."
        )

    return "\n".join(report)
