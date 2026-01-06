import numpy as np
import pandas as pd

#--- Function : premodeling_regression_check ---
def premodeling_regression_check(
    df: pd.DataFrame,
    target: str | None = None,
    corr_threshold: float = 0.7,
    top_outliers: int = 5,
    top_correlations: int = 10,
    min_unique_for_continuous: int = 10,
    iqr_multiplier: float = 1.5,
    final_assessment: bool = True
) -> str:
    """
    Expert-level pre-modeling validation for regression tasks.
    Returns a structured, readable diagnostic report.
    """

    sections = {}
    issues = set()

    #Missing values
    block = ["# Missing values"]
    missing = df.isnull().sum()
    missing = missing[missing > 0]

    if missing.empty:
        block.append("No missing values detected.")
    else:
        issues.add("missing")
        block.append("Missing values detected:")
        for col, cnt in missing.items():
            block.append(f"- {col}: {cnt}")

    sections["missing"] = block

    #Feature types
    block = ["# Feature types"]
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) == df.shape[1]:
        block.append("No non-numeric columns detected.")
    else:
        issues.add("types")
        non_numeric = sorted(set(df.columns) - set(numeric_cols))
        block.append("Non-numeric columns detected (require encoding):")
        for col in non_numeric:
            block.append(f"- {col}")

    sections["types"] = block

    #Feature variance
    block = ["# Feature variance"]
    constant_cols = [c for c in numeric_cols if df[c].nunique(dropna=True) <= 1]

    if not constant_cols:
        block.append("No constant columns detected.")
    else:
        issues.add("variance")
        block.append("Constant columns detected (should be removed):")
        for col in constant_cols:
            block.append(f"- {col}")

    sections["variance"] = block

    #Continuous feature detection
    continuous_original_cols = [
        c for c in numeric_cols
        if df[c].nunique(dropna=True) >= min_unique_for_continuous and "_PC" not in c
    ]

    continuous_pca_cols = [
        c for c in numeric_cols
        if df[c].nunique(dropna=True) >= min_unique_for_continuous and "_PC" in c
    ]

    def detect_outliers(cols):
        counts = {}
        total = 0
        for col in cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            lower = q1 - iqr_multiplier * iqr
            upper = q3 + iqr_multiplier * iqr
            count = ((df[col] < lower) | (df[col] > upper)).sum()
            if count > 0:
                counts[col] = int(count)
                total += int(count)
        return total, counts

    orig_total, orig_outliers = detect_outliers(continuous_original_cols)
    pca_total, pca_outliers = detect_outliers(continuous_pca_cols)

    block = ["# Outliers", "## Original continuous features"]
    if not orig_outliers:
        block.append("No significant outliers detected.")
    else:
        issues.add("outliers")
        sorted_out = sorted(orig_outliers.items(), key=lambda x: x[1], reverse=True)
        block.append(f"- Total outliers detected: {orig_total}")
        block.append("- Top contributing features:")
        for i, (col, cnt) in enumerate(sorted_out[:top_outliers], 1):
            block.append(f"  {i}. {col}: {cnt}")
        if len(sorted_out) > top_outliers:
            block.append(f"- {len(sorted_out) - top_outliers} additional features with minor outlier presence")

    block.append("## PCA-derived features")
    if not pca_outliers:
        block.append("No significant PCA-related outliers detected.")
    else:
        issues.add("outliers_pca")
        sorted_out = sorted(pca_outliers.items(), key=lambda x: x[1], reverse=True)
        block.append(f"- Total PCA outliers detected: {pca_total}")
        block.append("- Top contributing components:")
        for i, (col, cnt) in enumerate(sorted_out[:top_outliers], 1):
            block.append(f"  {i}. {col}: {cnt}")
        if len(sorted_out) > top_outliers:
            block.append(f"- {len(sorted_out) - top_outliers} additional PCA components with minor outlier presence")

    sections["outliers"] = block

    #Correlations
    block = [f"# High correlations (|r| ≥ {corr_threshold:.2f})"]
    corr_pairs = []

    if len(numeric_cols) >= 2:
        corr_matrix = df[numeric_cols].corr().abs()
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                val = corr_matrix.iloc[i, j]
                if val >= corr_threshold:
                    corr_pairs.append((corr_matrix.columns[j], corr_matrix.columns[i], float(val)))

    if not corr_pairs:
        block.append("No problematic correlations detected.")
    else:
        issues.add("correlations")
        corr_pairs = sorted(corr_pairs, key=lambda x: x[2], reverse=True)
        block.append("- Strongest correlations:")
        for i, (c1, c2, val) in enumerate(corr_pairs[:top_correlations], 1):
            block.append(f"  {i}. {c1} ↔ {c2}: {val:.3f}")
        if len(corr_pairs) > top_correlations:
            block.append(f"- {len(corr_pairs) - top_correlations} additional correlated pairs above threshold")

    sections["correlations"] = block

    #Target
    block = ["# Target validation"]
    if target is None:
        block.append("No target specified.")
    elif target not in df.columns:
        issues.add("target")
        block.append("Target column not found in dataset.")
    elif df[target].isnull().any():
        issues.add("target")
        block.append("Target contains missing values.")
    elif df[target].nunique() <= 1:
        issues.add("target")
        block.append("Target has zero variance.")
    else:
        block.append("No issues detected with target variable.")

    sections["target"] = block

    #Dataset size
    block = ["# Dataset size"]
    if df.shape[0] < 30:
        issues.add("size")
        block.append("Dataset may be too small for reliable regression.")
    else:
        block.append("No size-related risks detected.")

    sections["size"] = block

    #Final assessment
    final_block = ["# Final assessment"]
    if not issues:
        final_block.append("No structural issues detected. Data is ready for regression modeling.")
    else:
        final_block.append(
            "Dataset is usable for regression, but issues above should be reviewed "
            "depending on model choice (linear, regularized, tree-based, or neural)."
        )

    #Rendering logic
    if final_assessment:
        output = []
        for key in sections:
            output.extend(sections[key])
            output.append("")
        output.extend(final_block)
        return "\n".join(output).strip()

    if not issues:
        return final_block[1]

    output = []
    for key in sections:
        if key in issues or key == "outliers":
            output.extend(sections[key])
            output.append("")
    output.extend(final_block)

    return "\n".join(output).strip()
