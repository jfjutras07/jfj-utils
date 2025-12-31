import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import shapiro, normaltest, anderson, kstest, levene, bartlett, probplot
import math

def stats_diagnostics(df, numeric_cols=None, group_col=None, model=None, predictors=None):
    """
    Full diagnostics: Q-Q plots, residuals vs fitted (2x2 grid), normality and homogeneity tests.
    """
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include='number').columns.tolist()

    results = {}

    # --- 1. Q-Q plots ---
    n_cols = 2
    n_rows = int(np.ceil(len(numeric_cols)/2))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5*n_rows))
    axes = axes.flatten()
    for i, col in enumerate(numeric_cols):
        col_data = df[col].dropna()
        probplot(col_data, dist="norm", plot=axes[i])
        axes[i].set_title(f"Q-Q Plot of {col}")
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.show()

    # --- 2. Residuals vs fitted plots (2x2 grid) ---
    if model is not None and predictors is not None and len(predictors) >= 1:
        resid = getattr(model, 'resid')
        fitted = getattr(model, 'fittedvalues')

        n_plots = len(predictors)
        n_cols = 2
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
            axes[j].set_visible(False)  # case vide à droite si impair
        plt.tight_layout()
        plt.show()

    # --- 3. Normality tests ---
    normal_res = []
    for col in numeric_cols:
        s = df[col].dropna()
        n = len(s)
        mean = s.mean()
        std = s.std()
        shapiro_flag = dagostino_flag = anderson_flag = ks_flag = np.nan
        shapiro_p = dagostino_p = ad_stat = ks_p = np.nan

        if n <= 5000:
            stat, shapiro_p = shapiro(s)
            shapiro_flag = shapiro_p > 0.05
        if n > 20:
            stat, dagostino_p = normaltest(s)
            dagostino_flag = dagostino_p > 0.05
        ad_result = anderson(s)
        crit_val_5 = ad_result.critical_values[2]
        ad_stat = ad_result.statistic
        anderson_flag = ad_stat < crit_val_5
        _, ks_p = kstest(s, 'norm', args=(mean, std))
        ks_flag = ks_p > 0.05

        normal_res.append({
            "Column": col,
            "N": n,
            "Mean": round(mean,2),
            "Std": round(std,2),
            "Shapiro p": round(shapiro_p,4) if not np.isnan(shapiro_p) else np.nan,
            "Shapiro pass": "✔" if shapiro_flag else "✖",
            "D’Agostino p": round(dagostino_p,4) if not np.isnan(dagostino_p) else np.nan,
            "D’Agostino pass": "✔" if dagostino_flag else "✖",
            "Anderson stat": round(ad_stat,4),
            "Anderson pass": "✔" if anderson_flag else "✖",
            "KS p": round(ks_p,4),
            "KS pass": "✔" if ks_flag else "✖"
        })

    normality_df = pd.DataFrame(normal_res)
    results['normality'] = normality_df
    print("\n=== Normality Tests ===")
    display(normality_df)

    # --- 4. Homogeneity of variance ---
    if group_col is not None:
        hom_res = []
        df[group_col] = df[group_col].astype('category')
        for col in numeric_cols:
            groups_data = [df[df[group_col]==lvl][col].dropna() for lvl in df[group_col].cat.categories]
            levene_stat, levene_p = levene(*groups_data, center='median')
            bartlett_stat, bartlett_p = bartlett(*groups_data)
            hom_res.append({
                "Column": col,
                "Levene stat": round(levene_stat,4),
                "Levene p": round(levene_p,4),
                "Levene pass": "✔" if levene_p>0.05 else "✖",
                "Bartlett stat": round(bartlett_stat,4),
                "Bartlett p": round(bartlett_p,4),
                "Bartlett pass": "✔" if bartlett_p>0.05 else "✖"
            })
        homogeneity_df = pd.DataFrame(hom_res)
        results['homogeneity'] = homogeneity_df
        print("\n=== Homogeneity Tests ===")
        display(homogeneity_df)

    return results
