import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import shapiro, normaltest, anderson, kstest, levene, bartlett, probplot

def stats_diagnostics(df, numeric_cols=None, group_col=None, model=None, predictors=None):
    """
    Diagnostics: QQ plot | Residuals vs Fitted côte à côte, normality and homogeneity tables.
    Returns only DataFrames for normality and homogeneity.
    """
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include='number').columns.tolist()

    results = {}

    # --- QQ plot et Residuals vs Fitted côte à côte ---
    if model is not None:
        resid = getattr(model, 'resid')
        fitted = getattr(model, 'fittedvalues')
    for col in numeric_cols:
        col_data = df[col].dropna()
        fig, axes = plt.subplots(1, 2, figsize=(10,5))  # 1 ligne, 2 colonnes
        # QQ plot
        probplot(col_data, dist="norm", plot=axes[0])
        axes[0].set_title(f"Q-Q Plot of {col}")
        axes[0].set_xlabel("Theoretical Quantiles")
        axes[0].set_ylabel("Sample Quantiles")
        # Residuals vs fitted
        if model is not None:
            axes[1].scatter(fitted, resid, alpha=0.7)
            axes[1].axhline(0, color='red', linestyle='--')
            axes[1].set_xlabel('Fitted values')
            axes[1].set_ylabel('Residuals')
            axes[1].set_title(f'Residuals vs Fitted')
        plt.tight_layout()
        plt.show()

    # --- Normality tests ---
    normal_res = []
    for col in numeric_cols:
        s = df[col].dropna()
        n = len(s)
        mean = s.mean()
        std = s.std()
        shapiro_flag = dagostino_flag = anderson_flag = ks_flag = np.nan
        shapiro_p = dagostino_p = ks_p = np.nan

        if n <= 5000:
            _, shapiro_p = shapiro(s)
            shapiro_flag = shapiro_p > 0.05
        if n > 20:
            _, dagostino_p = normaltest(s)
            dagostino_flag = dagostino_p > 0.05
        ad_result = anderson(s)
        crit_val_5 = ad_result.critical_values[2]
        anderson_flag = ad_result.statistic < crit_val_5
        _, ks_p = kstest(s, 'norm', args=(mean,std))
        ks_flag = ks_p > 0.05

        normal_res.append({
            "Column": col,
            "N": n,
            "Mean": round(mean,2),
            "Std": round(std,2),
            "Shapiro p": round(shapiro_p,4) if shapiro_p is not None else np.nan,
            "Shapiro pass": shapiro_flag,
            "D’Agostino p": round(dagostino_p,4) if dagostino_p is not None else np.nan,
            "D’Agostino pass": dagostino_flag,
            "Anderson stat": round(ad_result.statistic,4),
            "Anderson pass": anderson_flag,
            "KS p": round(ks_p,4),
            "KS pass": ks_flag
        })
    results['normality'] = pd.DataFrame(normal_res)

    # --- Homogeneity ---
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
                "Levene pass": levene_p>0.05,
                "Bartlett stat": round(bartlett_stat,4),
                "Bartlett p": round(bartlett_p,4),
                "Bartlett pass": bartlett_p>0.05
            })
        results['homogeneity'] = pd.DataFrame(hom_res)

    return results
