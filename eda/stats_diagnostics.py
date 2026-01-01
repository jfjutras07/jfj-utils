import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_breuschpagan, het_white

#--- Function : stats_diagnostics ---
def stats_diagnostics(df, numeric_cols=None, group_col=None, model=None, predictors=None):

    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include='number').columns.tolist()

    point_color = 'steelblue'

    # ======================================================
    # Prepare data for plots
    # ======================================================
    if model is not None:
        resid = model.resid
        fitted = model.fittedvalues
        df_resid = pd.DataFrame({'Residuals': resid})
        cols_to_plot = ['Residuals']
    else:
        cols_to_plot = numeric_cols.copy()

    # ======================================================
    # Plots: QQ plots + Residuals vs Fitted
    # ======================================================
    n_plots = len(cols_to_plot) + (1 if model is not None else 0)
    n_cols = 2
    n_rows = int(np.ceil(n_plots / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5 * n_rows))
    axes = axes.flatten()

    #QQ plots
    for i, col in enumerate(cols_to_plot):
        series = df_resid[col] if model else df[col]
        stats.probplot(series.dropna(), dist="norm", plot=axes[i])
        for line in axes[i].get_lines():
            line.set_color(point_color)
        axes[i].set_title(f"Q-Q Plot: {col}")

    #Residuals vs Fitted
    if model is not None:
        j = len(cols_to_plot)
        axes[j].scatter(fitted, resid, alpha=0.7, color=point_color)
        axes[j].axhline(0, color='red', linestyle='--')
        axes[j].set_xlabel('Fitted values')
        axes[j].set_ylabel('Residuals')
        axes[j].set_title('Residuals vs Fitted')

    for k in range(n_plots, len(axes)):
        axes[k].axis('off')

    plt.tight_layout()
    plt.show()

    # ======================================================
    # Normality tests
    # ======================================================
    normal_res = []

    for col in cols_to_plot:
        s = (df_resid[col] if model else df[col]).dropna()
        n = len(s)
        mean, std = s.mean(), s.std()

        shapiro_p = stats.shapiro(s)[1] if n <= 5000 else np.nan
        dagostino_p = stats.normaltest(s)[1] if n > 20 else np.nan
        ad_stat = stats.anderson(s).statistic
        ks_p = stats.kstest(s, 'norm', args=(mean, std))[1]

        normal_res.append({
            'Column': col,
            'Shapiro': round(shapiro_p, 4) if not np.isnan(shapiro_p) else '-',
            'Dagostino': round(dagostino_p, 4) if not np.isnan(dagostino_p) else '-',
            'Anderson': round(ad_stat, 4),
            'KS': round(ks_p, 4)
        })

    normal_df = pd.DataFrame(normal_res)

    print("=== Normality Tests ===")
    display(
        normal_df.style
        .set_table_styles([
            {'selector': 'th', 'props': [('text-align', 'left')]},
            {'selector': 'td', 'props': [('text-align', 'left')]}
        ])
        .background_gradient(cmap='Blues', subset=normal_df.columns[1:])
    )

    # ======================================================
    # Heteroscedasticity tests 
    # ======================================================
    hetero_results = []

    #Group-based tests
    if group_col is not None:
        df[group_col] = df[group_col].astype('category')

        for col in numeric_cols:
            groups = [
                df[df[group_col] == lvl][col].dropna()
                for lvl in df[group_col].cat.categories
            ]

            hetero_results.append({
                'Test': f'Levene ({col})',
                'p-value': stats.levene(*groups, center='median')[1]
            })

            hetero_results.append({
                'Test': f'Bartlett ({col})',
                'p-value': stats.bartlett(*groups)[1]
            })

    #Model-based tests
    if model is not None:
        exog = model.model.exog
        bp_test = het_breuschpagan(model.resid, exog)
        white_test = het_white(model.resid, exog)

        hetero_results.extend([
            {'Test': 'Breusch-Pagan LM', 'p-value': bp_test[1]},
            {'Test': 'White LM', 'p-value': white_test[1]}
        ])

    hetero_df = (
        pd.DataFrame(hetero_results)
        .assign(**{'p-value': lambda x: x['p-value'].round(4)})
        .set_index('Test')
    )

    print("\n=== Heteroscedasticity Tests ===")
    display(
        hetero_df.style
        .set_table_styles([
            {'selector': 'th', 'props': [('text-align', 'left')]},
            {'selector': 'td', 'props': [('text-align', 'left')]}
        ])
        .background_gradient(cmap='Blues')
    )
