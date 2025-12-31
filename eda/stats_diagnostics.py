import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from scipy.stats import shapiro, normaltest, anderson, kstest, levene, bartlett, stats

#--- Function: stats_diagnostics ---
def stats_diagnostics(df, numeric_cols=None, group_col=None, model=None, predictors=None):
    """
    Perform a full set of diagnostics:

    1. Q-Q plots for numeric columns (2 per row)
    2. Residuals vs fitted plots (if model provided)
    3. Normality tests for numeric columns
    4. Homogeneity of variance across groups (if group_col provided)

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to analyze
    numeric_cols : list of str, optional
        Numeric columns to check. Defaults to all numeric columns.
    group_col : str, optional
        Categorical column for homogeneity tests.
    model : fitted model object, optional
        Model with residuals and fitted values.
    predictors : list of str, optional
        Predictor names for residual plots.

    Returns
    -------
    results : dict
        Dictionary containing:
        - 'normality': DataFrame with normality tests
        - 'homogeneity': dict with Levene and Bartlett results (if group_col)
    """
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include='number').columns.tolist()

    # --- 1. Q-Q plots ---
    n_cols = 2
    n_rows = math.ceil(len(numeric_cols)/n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5*n_rows))
    axes = axes.flatten()
    for i, col in enumerate(numeric_cols):
        col_data = df[col].dropna()
        stats.probplot(col_data, dist="norm", plot=axes[i])
        axes[i].set_title(f"Q-Q Plot of {col}")
        axes[i].set_xlabel("Theoretical Quantiles")
        axes[i].set_ylabel("Sample Quantiles")
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.show()

    # --- 2. Residuals vs fitted plots ---
    if model is not None:
        resid = getattr(model, 'resid')
        fitted = getattr(model, 'fittedvalues')
        if predictors is None:
            plt.figure(figsize=(8,6))
            plt.scatter(fitted, resid, alpha=0.7)
            plt.axhline(0, color='red', linestyle='--')
            plt.xlabel('Fitted values')
            plt.ylabel('Residuals')
            plt.title('Residuals vs Fitted Values')
            plt.show()
        else:
            n_plots = len(predictors)
            n_rows = math.ceil(n_plots / n_cols)
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
            axes = axes.flatten()
            for i, var in enumerate(predictors):
                axes[i].scatter(df[var], resid, alpha=0.7)
                axes[i].axhline(0, color='red', linestyle='--')
                axes[i].set_xlabel(var)
                axes[i].set_ylabel('Residuals')
                axes[i].set_title(f'Residuals vs {var}')
            for j in range(i+1, len(axes)):
                axes[j].set_visible(False)
            plt.tight_layout()
            plt.show()

    results = {}

    # --- 3. Normality tests ---
    normal_res = []
    for col in numeric_cols:
        s = df[col].dropna()
        n = len(s)
        col_mean = s.mean()
        col_std = s.std()
        shapiro_flag = np.nan
        dagostino_flag = np.nan
        anderson_flag = np.nan
        ks_flag = np.nan
        if n <= 5000:
            _, p = shapiro(s)
            shapiro_flag = p > 0.05
        if n > 20:
            _, p = normaltest(s)
            dagostino_flag = p > 0.05
        ad_result = anderson(s)
        crit_val_5 = ad_result.critical_values[2]
        ad_stat = ad_result.statistic
        anderson_flag = ad_stat < crit_val_5
        _, ks_p = kstest(s, 'norm', args=(col_mean, col_std))
        ks_flag = ks_p > 0.05
        normal_res.append({
            "Column": col,
            "N": n,
            "Mean": col_mean,
            "Std": col_std,
            "Shapiro-Wilk": shapiro_flag,
            "D’Agostino K²": dagostino_flag,
            "Anderson-Darling (5%)": anderson_flag,
            "Kolmogorov-Smirnov": ks_flag
        })
    results['normality'] = pd.DataFrame(normal_res)

    # --- 4. Homogeneity of variance ---
    if group_col is not None:
        hom_res = {}
        for col in numeric_cols:
            df[group_col] = df[group_col].astype('category')
            groups_data = [df[df[group_col]==lvl][col].dropna() for lvl in df[group_col].cat.categories]
            levene_stat, levene_p = levene(*groups_data, center='median')
            bartlett_stat, bartlett_p = bartlett(*groups_data)
            hom_res[col] = {
                'levene_stat': levene_stat,
                'levene_p': levene_p,
                'bartlett_stat': bartlett_stat,
                'bartlett_p': bartlett_p
            }
        results['homogeneity'] = hom_res

    return results
