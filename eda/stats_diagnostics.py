import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from scipy.stats import shapiro, normaltest, anderson, kstest, levene, bartlett, stats
from tabulate import tabulate

def stats_diagnostics(df, numeric_cols=None, group_col=None, model=None, predictors=None):
    """
    Full diagnostics with:
    1. Q-Q plots for numeric columns
    2. Residuals vs fitted plots
    3. Normality tests
    4. Homogeneity tests
    5. Beautiful structured tables

    Parameters
    ----------
    df : pd.DataFrame
    numeric_cols : list of str, optional
    group_col : str, optional
    model : fitted model object, optional
    predictors : list of str, optional

    Returns
    -------
    results : dict
        'normality': DataFrame
        'homogeneity': DataFrame (if group_col)
    """
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include='number').columns.tolist()

    # --- 1. Q-Q plots & Residuals side by side ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # QQ plot
    col = numeric_cols[0]  # first numeric column for QQ plot
    col_data = df[col].dropna()
    stats.probplot(col_data, dist="norm", plot=axes[0])
    axes[0].set_title(f"Q-Q Plot of {col}")
    
    # Residuals vs fitted
    if model is not None:
        resid = getattr(model, 'resid')
        fitted = getattr(model, 'fittedvalues')
        axes[1].scatter(fitted, resid, alpha=0.7)
        axes[1].axhline(0, color='red', linestyle='--')
        axes[1].set_xlabel('Fitted values')
        axes[1].set_ylabel('Residuals')
        axes[1].set_title('Residuals vs Fitted')
    
    plt.tight_layout()
    plt.show()

    results = {}

    # --- 2. Normality tests ---
    normal_res = []
    for col in numeric_cols:
        s = df[col].dropna()
        n = len(s)
        col_mean = s.mean()
        col_std = s.std()
        shapiro_p = np.nan
        shapiro_pass = np.nan
        dagostino_p = np.nan
        dagostino_pass = np.nan
        ad_stat = np.nan
        ad_pass = np.nan
        ks_p = np.nan
        ks_pass = np.nan

        if n <= 5000:
            _, shapiro_p = shapiro(s)
            shapiro_pass = shapiro_p > 0.05
        if n > 20:
            _, dagostino_p = normaltest(s)
            dagostino_pass = dagostino_p > 0.05
        ad_result = anderson(s)
        crit_val_5 = ad_result.critical_values[2]
        ad_stat = ad_result.statistic
        ad_pass = ad_stat < crit_val_5
        _, ks_p = kstest(s, 'norm', args=(col_mean, col_std))
        ks_pass = ks_p > 0.05

        normal_res.append({
            "Column": col,
            "N": n,
            "Mean": round(col_mean,2),
            "Std": round(col_std,2),
            "Shapiro p": round(shapiro_p,4) if not np.isnan(shapiro_p) else np.nan,
            "Shapiro pass": shapiro_pass,
            "D’Agostino p": round(dagostino_p,4) if not np.isnan(dagostino_p) else np.nan,
            "D’Agostino pass": dagostino_pass,
            "Anderson stat": round(ad_stat,4),
            "Anderson pass": ad_pass,
            "KS p": round(ks_p,4),
            "KS pass": ks_pass
        })
    normal_df = pd.DataFrame(normal_res)
    results['normality'] = normal_df
    print("=== Normality Tests ===")
    print(tabulate(normal_df, headers='keys', tablefmt='grid', showindex=False))

    # --- 3. Homogeneity ---
    if group_col is not None:
        hom_res = []
        for col in numeric_cols:
            df[group_col] = df[group_col].astype('category')
            groups_data = [df[df[group_col]==lvl][col].dropna() for lvl in df[group_col].cat.categories]
            levene_stat, levene_p = levene(*groups_data, center='median')
            bartlett_stat, bartlett_p = bartlett(*groups_data)
            hom_res.append({
                "Column": col,
                "Levene stat": round(levene_stat,4),
                "Levene p": round(levene_p,4),
                "Levene pass": levene_p>0.05,
                "Bartlett stat": round(bartlett_stat,4),
                "Bartlett p": round(bartlett_p,4),
                "Bartlett pass": bartlett_p>0.05
            })
        hom_df = pd.DataFrame(hom_res)
        results['homogeneity'] = hom_df
        print("\n=== Homogeneity Tests ===")
        print(tabulate(hom_df, headers='keys', tablefmt='grid', showindex=False))

    return results
