import pandas as pd
import numpy as np

def premodeling_regression_check(
    df: pd.DataFrame,
    target: str | None = None,
    corr_threshold: float = 0.7
) -> str:
    """
    Generic pre-modeling diagnostic for regression tasks.
    Designed to be model-agnostic (linear, regularized, trees, ensembles, deep learning).

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to analyze.
    target : str, optional
        Target column name.
    corr_threshold : float
        Absolute correlation threshold for multicollinearity warning.

    Returns
    -------
    str
        Markdown-style diagnostic report.
    """

    report = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # ------------------------------------------------------------------
    # Missing values
    # ------------------------------------------------------------------
    na_cols = df.isna().sum()
    na_cols = na_cols[na_cols > 0]

    if not na_cols.empty:
        report.append(
            "# Missing values detected\n"
            + "\n".join([f"- {col}: {int(count)}" for col, count in na_cols.items()])
        )
    else:
        report.append("# Missing values\nNo missing values detected.")

    # ------------------------------------------------------------------
    # Non-numeric features
    # ------------------------------------------------------------------
    non_numeric_cols = df.columns.difference(numeric_cols)

    if len(non_numeric_cols) > 0:
        report.append(
            "# Non-numeric columns detected\n"
            + "\n".join([f"- {col}" for col in non_numeric_cols])
        )
    else:
        report.append("# Feature types\nNo non-numeric columns detected.")

    # ------------------------------------------------------------------
    # Constant or quasi-constant columns
    # ------------------------------------------------------------------
    constant_cols = [col for col in df.columns if df[col].nunique(dropna=False) <= 1]

    if constant_cols:
        report.append(
            "# Constant columns detected\n"
            + "\n".join([f"- {col}" for col in constant_cols])
        )
    else:
        report.append("# Feature variance\nNo constant columns detected.")

    # ------------------------------------------------------------------
    # Multicollinearity (correlation-based)
    # ------------------------------------------------------------------
    high_corr_pairs = []
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr().abs()
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] >= corr_threshold:
                    high_corr_pairs.append(
                        (
                            corr_matrix.columns[i],
                            corr_matrix.columns[j],
                            float(corr_matrix.iloc[i, j])
                        )
                    )

    if high_corr_pairs:
        report.append(
            "# High correlations detected (risk for linear and regularized models)\n"
            + "\n".join(
                [f"- {a} ↔ {b}: {v:.3f}" for a, b, v in high_corr_pairs]
            )
        )
    else:
        report.append("# Multicollinearity\nNo correlations above threshold detected.")

    # ------------------------------------------------------------------
    # Target validation
    # ------------------------------------------------------------------
    if target:
        if target not in df.columns:
            report.append(
                f"# Target validation\nTarget column '{target}' not found."
            )
        else:
            target_issues = []

            if df[target].isna().any():
                target_issues.append("contains missing values")

            if df[target].nunique() <= 1:
                target_issues.append("is constant")

            if not pd.api.types.is_numeric_dtype(df[target]):
                target_issues.append("is not numeric")

            if target_issues:
                report.append(
                    "# Target validation issues\n"
                    + "\n".join([f"- Target {issue}" for issue in target_issues])
                )
            else:
                report.append("# Target validation\nNo issues detected with target variable.")
    else:
        report.append("# Target validation\nNo target provided (skipped).")

    # ------------------------------------------------------------------
    # Dataset size sanity check
    # ------------------------------------------------------------------
    if len(numeric_cols) > 0 and len(df) < 5 * len(numeric_cols):
        report.append(
            "# Dataset size warning\n"
            f"Rows ({len(df)}) < 5 × numeric features ({len(numeric_cols)}). "
            "Risk of overfitting for parametric models."
        )
    else:
        report.append("# Dataset size\nNo size-related risks detected.")

    # ------------------------------------------------------------------
    # Final verdict
    # ------------------------------------------------------------------
    problems = [
        section for section in report
        if "detected" in section.lower() or "issues" in section.lower() or "warning" in section.lower()
    ]

    if not problems:
        report.append(
            "# Final assessment\n"
            "No missing values.\n"
            "No non-numeric features.\n"
            "No constant features.\n"
            "No harmful multicollinearity.\n"
            "No target-related issues.\n"
            "No dataset size risks.\n"
            "Data is ready for regression modeling."
        )

    return "\n\n".join(report)
