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

    # --- Plots ---
    point_color = 'steelblue'
    
    # QQ-plots pour toutes les colonnes numériques ou résidus
    cols_to_plot = numeric_cols.copy()
    if model is not None:
        resid = model.resid
        fitted = model.fittedvalues
        df_resid = pd.DataFrame({'Residuals': resid})
        cols_to_plot = ['Residuals']
    
    for col in cols_to_plot:
        plt.figure(figsize=(6,4))
        stats.probplot(df_resid[col] if model else df[col], dist="norm", plot=plt)
        plt.title(f"Q-Q Plot: {col}")
        plt.show()
    
    # Residuals vs Fitted uniquement si modèle fourni
    if model is not None:
        plt.figure(figsize=(6,4))
        plt.scatter(fitted, resid, alpha=0.7, color=point_color)
        plt.axhline(0, color='red', linestyle='--')
        plt.xlabel('Fitted values')
        plt.ylabel('Residuals')
        plt.title('Residuals vs Fitted')
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
    hom_df = pd.DataFrame()
    if group_col is not None:
        # Levene & Bartlett
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
        print("\n=== Homogeneity Tests (Levene & Bartlett) ===")
        display(hom_df.style.background_gradient(cmap='Blues', subset=hom_df.columns[1:]))

    elif model is not None:
        # Breusch-Pagan et White pour OLS
        exog = model.model.exog
        bp_test = het_breuschpagan(model.resid, exog)
        white_test = het_white(model.resid, exog)
        tests = pd.DataFrame({
            'Test': ['Breusch-Pagan LM', 'Breusch-Pagan p-value', 
                     'White LM', 'White p-value'],
            'Value': [bp_test[0], bp_test[1], white_test[0], white_test[1]]
        })
        print("\n=== Heteroscedasticity Tests (Breusch-Pagan & White) ===")
        display(tests.style.background_gradient(cmap='Oranges', subset=['Value']))
