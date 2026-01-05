import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, het_white

UNIFORM_BLUE = 'steelblue'
LINE_RED = 'red'

#--- Function : stats_diagnostics ---
def stats_diagnostics(df, numeric_cols=None, group_col=None, model=None, predictors=None):
    """
    Perform basic statistical diagnostics for numeric data or model residuals.

    Features:
    ---------
    1. Q-Q plots for checking normality of each numeric variable or model residuals.
    2. Residuals vs Fitted plot if a model is provided (check for heteroscedasticity).
    3. Normality tests:
       - Shapiro-Wilk (small to medium samples)
       - D'Agostino's K^2 (medium to large samples)
       - Anderson-Darling
       - Kolmogorov-Smirnov
    4. Variance homogeneity / heteroscedasticity:
       - Levene and Bartlett tests if grouping variable is provided
       - Breusch-Pagan and White tests if a model is provided and supported
    5. Influence diagnostics (Cook's distance) if model supports it (OLS only).

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the data.
    numeric_cols : list, optional
        List of numeric columns to analyze. Defaults to all numeric columns.
    group_col : str, optional
        Categorical column for group-wise variance tests.
    model : statsmodels regression result, optional
        Fitted model object to analyze residuals and perform heteroscedasticity tests.
    predictors : list, optional
        List of predictors in the model (currently not used).

    Returns:
    --------
    Displays plots and styled DataFrames of normality and variance tests.
    Does not return values.
    """
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include='number').columns.tolist()

    # Prepare columns for plotting
    cols_to_plot = numeric_cols.copy()
    if model is not None:
        try:
            resid = model.resid
            fitted = model.fittedvalues
            df_resid = pd.DataFrame({'Residuals': resid})
            cols_to_plot = ['Residuals']
        except AttributeError:
            print("Warning: Model has no residuals/fittedvalues. Only numeric columns will be plotted.")
            model = None

    # QQ plots and Residuals vs Fitted
    n_plots = len(cols_to_plot) + (1 if model is not None else 0)
    n_cols = 2
    n_rows = int(np.ceil(n_plots / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(cols_to_plot):
        s = df_resid[col] if model else df[col]
        stats.probplot(s.dropna(), dist="norm", plot=axes[i])
        lines = axes[i].get_lines()
        if len(lines) >= 2:
            lines[0].set_color(UNIFORM_BLUE)
            lines[1].set_color(LINE_RED)
        axes[i].set_title(f"Q-Q Plot: {col}")

    if model is not None:
        j = len(cols_to_plot)
        axes[j].scatter(fitted, resid, alpha=0.7, color=UNIFORM_BLUE)
        axes[j].axhline(0, color=LINE_RED, linestyle='--')
        axes[j].set_xlabel('Fitted values')
        axes[j].set_ylabel('Residuals')
        axes[j].set_title('Residuals vs Fitted')

        # Cook's distance only if available
        if hasattr(model, "get_influence"):
            influence = model.get_influence()
            cooks_d = influence.cooks_distance[0]
            plt.figure(figsize=(10,5))
            plt.stem(cooks_d, linefmt='grey', markerfmt='D', basefmt=' ')
            plt.axhline(4/len(cooks_d), color=LINE_RED, linestyle='--')
            plt.xlabel('Observation index')
            plt.ylabel("Cook's distance")
            plt.title("Influential Observations (Cook's distance)")
            plt.show()
        else:
            print("Cook's distance not available for this model type. Skipping.")

    for k in range(n_plots, len(axes)):
        axes[k].axis('off')

    plt.tight_layout()
    plt.show()

    # Normality tests
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
            'Variable': col,
            'Shapiro': round(shapiro_p, 4) if shapiro_p != '-' else '-',
            'Dagostino': round(dagostino_p, 4) if dagostino_p != '-' else '-',
            'Anderson': round(ad_stat, 4),
            'KS': round(ks_p, 4)
        })
    normal_df = pd.DataFrame(normal_res)
    print("=== Normality Tests ===")
    display(
        normal_df.style.background_gradient(
            cmap='Blues',
            subset=normal_df.columns[1:]
        )
    )

    # Variance homogeneity / heteroscedasticity tests
    print("\n=== Variance Homogeneity / Heteroscedasticity Tests ===")
    if group_col is not None:
        df[group_col] = df[group_col].astype('category')
        levene_p = []
        bartlett_p = []
        for col in numeric_cols:
            groups = [
                df[df[group_col] == lvl][col].dropna()
                for lvl in df[group_col].cat.categories
            ]
            levene_p.append(stats.levene(*groups, center='median')[1])
            bartlett_p.append(stats.bartlett(*groups)[1])
        hetero_df = pd.DataFrame(
            {'Levene': np.round(levene_p, 4),
             'Bartlett': np.round(bartlett_p, 4)},
            index=numeric_cols
        )
        display(hetero_df.style.background_gradient(cmap='Blues'))
    elif model is not None:
        # Only perform BP and White if model has exog
        if hasattr(model.model, "exog"):
            exog = model.model.exog
            bp_test = het_breuschpagan(model.resid, exog)
            white_test = het_white(model.resid, exog)
            hetero_df = pd.DataFrame(
                {'Breusch-Pagan LM':[round(bp_test[1],4)],
                 'White LM':[round(white_test[1],4)]},
                index=['p-value']
            )
            display(hetero_df.style.background_gradient(cmap='Blues'))
        else:
            print("Breusch-Pagan / White tests not available for this model type.")
