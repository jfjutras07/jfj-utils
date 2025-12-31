import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from tabulate import tabulate

def stats_diagnostics(df, numeric_cols=None, group_col=None, model=None, predictors=None):
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include='number').columns.tolist()

    results = {}

    # --- QQ plots and residuals vs fitted ---
    if model is not None:
        if hasattr(model, 'resid') and hasattr(model, 'fittedvalues'):
            resid = model.resid
            fitted = model.fittedvalues
        else:
            raise ValueError("Model type not recognized. Must have 'resid' and 'fittedvalues'.")

        # Plot
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
        df_resid = pd.DataFrame({'Residuals': resid})
    else:
        numeric_cols_to_test = numeric_cols
        df_resid = df.copy()

    # --- Normality tests ---
    normal_res = []
    for col in numeric_cols_to_test:
        s = df_resid[col].dropna()
        n = len(s)
        col_mean, col_std = s.mean(), s.std()

        shapiro_p = stats.shapiro(s)[1] if n<=5000 else np.nan
        dagostino_p = stats.normaltest(s)[1] if n>20 else np.nan
        ad_stat = stats.anderson(s).statistic
        crit_val_5 = stats.anderson(s).critical_values[2]
        ad_pass = ad_stat < crit_val_5
        ks_p = stats.kstest(s, 'norm', args=(col_mean, col_std))[1]

        normal_res.append({
            'Column': col,
            'Shapiro': round(shapiro_p,4) if not np.isnan(shapiro_p) else '-',
            'Shapiro pass': 'Yes' if shapiro_p>0.05 else ('No' if not np.isnan(shapiro_p) else '-'),
            'Dagostino': round(dagostino_p,4) if not np.isnan(dagostino_p) else '-',
            'Dagostino pass': 'Yes' if dagostino_p>0.05 else ('No' if not np.isnan(dagostino_p) else '-'),
            'Anderson': round(ad_stat,4),
            'Anderson pass': 'Yes' if ad_pass else 'No',
            'KS': round(ks_p,4),
            'KS pass': 'Yes' if ks_p>0.05 else 'No'
        })

    # --- Homogeneity ---
    hom_res = []
    if group_col is not None:
        df[group_col] = df[group_col].astype('category')
        for col in numeric_cols:
            groups_data = [df[df[group_col]==lvl][col].dropna() for lvl in df[group_col].cat.categories]
            levene_stat, levene_p = stats.levene(*groups_data, center='median')
            bartlett_stat, bartlett_p = stats.bartlett(*groups_data)
            hom_res.append({
                'Column': col,
                'Levene': round(levene_p,4),
                'Levene pass': 'Yes' if levene_p>0.05 else 'No',
                'Bartlett': round(bartlett_p,4),
                'Bartlett pass': 'Yes' if bartlett_p>0.05 else 'No'
            })

    # --- Combine all results in a single table ---
    final_table = pd.DataFrame(normal_res)
    if hom_res:
        hom_df = pd.DataFrame(hom_res)
        final_table = final_table.merge(hom_df, on='Column', how='left')

    results['final_table'] = final_table
    print(tabulate(final_table, headers='keys', tablefmt='grid', showindex=False))

    return results
