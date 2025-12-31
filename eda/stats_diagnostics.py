import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from scipy.stats import shapiro, normaltest, anderson, kstest, levene, bartlett, probplot

#--- Function: stats_diagnostics ---
def stats_diagnostics(df, numeric_cols=None, group_col=None, model=None, predictors=None):
    """
    Perform full diagnostics: Q-Q plots, residuals vs fitted, normality and homogeneity tests.
    """
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include='number').columns.tolist()

    results = {}

    # --- 1. Q-Q plots ---
    n_cols = 2
    n_rows = math.ceil(len(numeric_cols)/n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5*n_rows))
    axes = axes.flatten()
    for i, col in enumerate(numeric_cols):
        col_data = df[col].dropna()
        probplot(col_data, dist="norm", plot=axes[i])
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
        if predictors is None or len(predictors) == 0:
            plt.figure(figsize=(8,6))
            plt.scatter(fitted, resid, alpha=0.7)
            plt.axhline(0, color='red', linestyle='--')
            plt.xlabel('Fitted values')
            plt.ylabel('Residuals')
            plt.title('Residuals vs Fitted')
            plt.show()
        else:
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
                axes[j].set_visible(False)
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
            "Mean": mean,
            "Std": std,
            "Shapiro-Wilk p": shapiro_p,
            "Shapiro-Wilk pass": shapiro_flag,
            "D’Agostino K² p": dagostino_p,
            "D’Agostino pass": dagostino_flag,
            "Anderson-Darling stat": ad_stat,
            "Anderson-Darling pass": anderson_flag,
            "Kolmogorov-Smirnov p": ks_p,
            "KS pass": ks_flag
        })

    normality_df = pd.DataFrame(normal_res)
    results['normality'] = normality_df
    print("\n=== Normality Tests ===")
    print(normality_df.to_string(index=False))

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
                "Levene stat": levene_stat,
                "Levene p": levene_p,
                "Levene pass": levene_p>0.05,
                "Bartlett stat": bartlett_stat,
                "Bartlett p": bartlett_p,
                "Bartlett pass": bartlett_p>0.05
            })
        homogeneity_df = pd.DataFrame(hom_res)
        results['homogeneity'] = homogeneity_df
        print("\n=== Homogeneity Tests ===")
        print(homogeneity_df.to_string(index=False))

    return results
