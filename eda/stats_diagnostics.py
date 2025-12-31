import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

def stats_diagnostics_clean(df, numeric_cols=None, group_col=None, model=None):
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include='number').columns.tolist()

    # --- Graphiques ---
    if model is not None:
        resid = model.resid
        fitted = model.fittedvalues
        fig, axes = plt.subplots(1, 2, figsize=(12,5))
        stats.probplot(resid, dist="norm", plot=axes[0])
        axes[0].set_title('Q-Q Plot of Residuals')
        axes[1].scatter(fitted, resid, alpha=0.7)
        axes[1].axhline(0, color='red', linestyle='--')
        axes[1].set_xlabel('Fitted values')
        axes[1].set_ylabel('Residuals')
        axes[1].set_title('Residuals vs Fitted')
        plt.tight_layout()
        plt.show()
        numeric_cols_to_test = ['Residuals']
        df_test = pd.DataFrame({'Residuals': resid})
    else:
        numeric_cols_to_test = numeric_cols
        df_test = df.copy()

    # --- Normalité ---
    normal_res = []
    for col in numeric_cols_to_test:
        s = df_test[col].dropna()
        n = len(s)
        mean, std = s.mean(), s.std()
        shapiro_p = stats.shapiro(s)[1] if n <= 5000 else '-'
        dagostino_p = stats.normaltest(s)[1] if n > 20 else '-'
        ad_stat = stats.anderson(s).statistic
        ks_p = stats.kstest(s, 'norm', args=(mean, std))[1]
        normal_res.append({
            'Column': col,
            'Shapiro': round(shapiro_p,4) if shapiro_p != '-' else '-',
            'Dagostino': round(dagostino_p,4) if dagostino_p != '-' else '-',
            'Anderson': round(ad_stat,4),
            'KS': round(ks_p,4)
        })
    normal_df = pd.DataFrame(normal_res)

    # --- Homogénéité ---
    hom_df = pd.DataFrame()
    if group_col is not None:
        hom_res = []
        df[group_col] = df[group_col].astype('category')
        for col in numeric_cols:
            groups = [df[df[group_col]==lvl][col].dropna() for lvl in df[group_col].cat.categories]
            levene_p = stats.levene(*groups, center='median')[1]
            bartlett_p = stats.bartlett(*groups)[1]
            hom_res.append({
                'Column': col,
                'Levene': round(levene_p,4),
                'Bartlett': round(bartlett_p,4)
            })
        hom_df = pd.DataFrame(hom_res)

    # --- Affichage propre ---
    print("=== Normality Tests ===")
    print(normal_df.to_string(index=False))
    if not hom_df.empty:
        print("\n=== Homogeneity Tests ===")
        print(hom_df.to_string(index=False))

    return normal_df, hom_df
