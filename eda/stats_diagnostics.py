import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_breuschpagan, het_white

def stats_diagnostics(df, numeric_cols=None, group_col=None, model=None, predictors=None):
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include='number').columns.tolist()

    point_color = 'steelblue'
    
    # --- Prepare columns for plotting ---
    cols_to_plot = numeric_cols.copy()
    if model is not None:
        resid = model.resid
        fitted = model.fittedvalues
        df_resid = pd.DataFrame({'Residuals': resid})
        cols_to_plot = ['Residuals']

    # --- Plots: QQ plots and Residuals vs Fitted ---
    n_plots = len(cols_to_plot) + (1 if model is not None else 0)  # extra plot for Residuals vs Fitted
    n_cols = 2
    n_rows = int(np.ceil(n_plots / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5*n_rows))
    axes = axes.flatten()
    
    # QQ-plots
    for i, col in enumerate(cols_to_plot):
        stats.probplot(df_resid[col] if model else df[col], dist="norm", plot=axes[i])
        for line in axes[i].get_lines():
            line.set_color(point_color)
        axes[i].set_title(f"Q-Q Plot: {col}")
    
    # Residuals vs Fitted
    if model is not None:
        j = len(cols_to_plot)
        axes[j].scatter(fitted, resid, alpha=0.7, color=point_color)
        axes[j].axhline(0, color='red', linestyle='--')
        axes[j].set_xlabel('Fitted values')
        axes[j].set_ylabel('Residuals')
        axes[j].set_title('Residuals vs Fitted')
    
    # Hide unused subplots
    for k in range(n_plots, len(axes)):
        axes[k].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # --- Normality tests ---
    normal_res = []
    for col in cols_to_plot:
        s = df_resid[col] if model else df[col]
        s = s.dropna()
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
    print("=== Normality Tests ===")
    display(normal_df.style.background_gradient(cmap='Blues', subset=normal_df.columns[1:]))

    # --- Homogeneity / Heteroscedasticity tests ---
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
        print("\n=== Homogeneity Tests ===")
        display(hom_df.style.background_gradient(cmap='Blues', subset=hom_df.columns[1:]))
    elif model is not None:
        exog = model.model.exog
        bp_test = het_breuschpagan(model.resid, exog)
        white_test = het_white(model.resid, exog)
        hetero_df = pd.DataFrame([{
            'Test': 'Breusch-Pagan LM',
            'p-value': round(bp_test[1],4)
        },{
            'Test': 'White LM',
            'p-value': round(white_test[1],4)
        }])
        print("\n=== Heteroscedasticity Tests ===")
        display(hetero_df.style.background_gradient(cmap='Blues', subset=['p-value']))
