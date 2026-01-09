import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import lifelines
from lifelines import CoxPHFitter

#---Function:cox_regression---
def cox_regression(df, duration_col, event_col, covariates):
    """
    Perform Cox Proportional Hazards regression for survival/time-to-event data.

    When to use:
    - Outcome is time-to-event (duration) and event indicator (1=event, 0=censored).
    - Estimate hazard ratios for predictors (covariates).
    - Useful in medical studies, reliability analysis, or any survival analysis.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset containing duration, event, and covariates.
    duration_col : str
        Column representing time to event or censoring.
    event_col : str
        Column representing event indicator (1=event occurred, 0=censored).
    covariates : list of str
        List of predictor variables (numeric or categorical).

    Returns:
    --------
    model : lifelines CoxPHFitter
        Fitted Cox Proportional Hazards model.
    """
    if not covariates:
        raise ValueError("At least one covariate must be provided.")

    cph = CoxPHFitter()
    cph.fit(df[[duration_col, event_col] + covariates], duration_col=duration_col, event_col=event_col)
    cph.print_summary()
    return cph

#---Function:gamma_regression---
def gamma_regression(df, outcome, predictors, include_interactions=False, link='log'):
    """
    Perform Gamma regression for positive continuous skewed data.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset containing outcome and predictors.
    outcome : str
        Dependent variable.
    predictors : list of str
        List of predictor variables.
    include_interactions : bool, default False
        Whether to include all pairwise interactions.
    link : str, default 'log'
        Link function ('log' or 'identity').

    Returns:
    --------
    model : statsmodels GLMResults
        Fitted Gamma regression model.
    """
    if not predictors:
        raise ValueError("At least one predictor must be provided.")

    formula_terms = [f"C({v})" if df[v].dtype.name == 'category' or df[v].dtype == object else v for v in predictors]
    sep = " * " if include_interactions else " + "
    formula = f"{outcome} ~ " + sep.join(formula_terms)

    link_func = sm.families.links.log() if link=='log' else sm.families.links.identity()
    model = smf.glm(formula=formula, data=df, family=sm.families.Gamma(link=link_func)).fit()
    print(model.summary())
    return model

#---Function:linear_mixed_model---
def linear_mixed_model(df, fixed_effects, outcome, random_effect, include_interactions=False):
    """
    Perform a Linear Mixed Model (LMM) regression.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset containing outcome, fixed effects, and grouping variable for random effect.
    fixed_effects : list of str
        List of fixed-effect predictor variables.
    outcome : str
        Dependent variable.
    random_effect : str
        Column name for random effect (grouping variable).
    include_interactions : bool, default False
        Whether to include all pairwise interactions between fixed effects.

    Returns:
    --------
    model_fit : MixedLMResults
        Fitted LMM.
    """
    if not fixed_effects:
        raise ValueError("At least one fixed effect must be provided.")

    formula_terms = [f"C({v})" if df[v].dtype.name == 'category' or df[v].dtype == object else v for v in fixed_effects]
    sep = " * " if include_interactions else " + "
    formula = f"{outcome} ~ {sep.join(formula_terms)}"
    
    model = smf.mixedlm(formula=formula, data=df, groups=df[random_effect])
    model_fit = model.fit(reml=True)
    print(model_fit.summary())
    return model_fit

#---Function:linear_regression---
def linear_regression(df, outcome, predictors, include_interactions=False):
    """
    Perform a multiple linear regression with optional interactions.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset containing predictors and outcome.
    outcome : str
        Dependent variable.
    predictors : list of str
        List of independent variables (numeric or categorical).
    include_interactions : bool, default False
        Whether to include all pairwise interactions between predictors.

    Returns:
    --------
    model : statsmodels RegressionResults
        Fitted linear regression model.
    """
    if not predictors:
        raise ValueError("At least one predictor must be provided.")

    formula_terms = [f"C({v})" if df[v].dtype.name == 'category' or df[v].dtype == object else v for v in predictors]
    sep = " * " if include_interactions else " + "
    formula = f"{outcome} ~ " + sep.join(formula_terms)

    model = smf.ols(formula=formula, data=df).fit()
    print(model.summary())
    return model

#---Function:negative_binomial_regression---
def negative_binomial_regression(df, outcome, predictors, include_interactions=False):
    """
    Perform a Negative Binomial regression for count data with overdispersion.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset containing predictors and outcome.
    outcome : str
        Dependent variable representing counts.
    predictors : list of str
        List of independent variables.
    include_interactions : bool, default False
        Whether to include all pairwise interactions.

    Returns:
    --------
    model : statsmodels RegressionResults
        Fitted Negative Binomial regression model.
    """
    if not predictors:
        raise ValueError("At least one predictor must be provided.")

    formula_terms = [f"C({v})" if df[v].dtype.name == 'category' or df[v].dtype == object else v for v in predictors]
    sep = " * " if include_interactions else " + "
    formula = f"{outcome} ~ " + sep.join(formula_terms)

    #Uses log link by default as it is standard for Negative Binomial
    model = smf.glm(formula=formula, data=df, family=sm.families.NegativeBinomial()).fit()
    print(model.summary())
    return model
    
#---Function:poisson_regression---
def poisson_regression(df, outcome, predictors, include_interactions=False):
    """
    Perform a Poisson regression for count data.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset containing predictors and outcome.
    outcome : str
        Dependent variable representing counts (non-negative integers).
    predictors : list of str
        List of independent numeric variables.
    include_interactions : bool, default False
        Whether to include all pairwise interactions between predictors.

    Returns:
    --------
    model : statsmodels RegressionResults
        Fitted Poisson regression model.
    """
    if not predictors:
        raise ValueError("At least one predictor must be provided.")

    formula_terms = [f"C({v})" if df[v].dtype.name == 'category' or df[v].dtype == object else v for v in predictors]
    sep = " * " if include_interactions else " + "
    formula = f"{outcome} ~ " + sep.join(formula_terms)

    model = smf.poisson(formula=formula, data=df).fit()
    print(model.summary())
    return model

#---Function:polynomial_regression---
def polynomial_regression(df, outcome, predictor, max_degree=2):
    """
    Perform polynomial regression up to a specified degree.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset containing predictor and outcome.
    outcome : str
        Dependent variable.
    predictor : str
        Independent numeric variable.
    max_degree : int, default=2
        Maximum polynomial degree (>=1).

    Returns:
    --------
    model : statsmodels RegressionResults
        Fitted polynomial regression model.
    """
    if max_degree < 1:
        raise ValueError("max_degree must be at least 1.")

    poly_terms = [predictor] + [f"I({predictor}**{d})" for d in range(2, max_degree + 1)]
    formula = f"{outcome} ~ " + " + ".join(poly_terms)

    model = smf.ols(formula=formula, data=df).fit()
    print(model.summary())
    return model

#---Function:quantile_regression---
def quantile_regression(df, outcome, predictor, quantile=0.5):
    """
    Perform quantile regression (median or other quantiles).

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset containing predictor and outcome.
    outcome : str
        Dependent variable.
    predictor : str
        Independent variable.
    quantile : float (0 < quantile < 1), default 0.5
        Quantile to estimate.

    Returns:
    --------
    model : statsmodels RegressionResults
        Fitted quantile regression model.
    """
    formula = f"{outcome} ~ {predictor}"
    model = smf.quantreg(formula=formula, data=df).fit(q=quantile)
    print(model.summary())
    return model

#---Function:robust_regression---
def robust_regression(df, outcome, predictors, include_interactions=False, estimator=sm.robust.norms.HuberT):
    """
    Perform a multiple robust linear regression using RLM (Robust Linear Model) 
    with optional interactions.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset containing predictors and outcome.
    outcome : str
        Dependent variable (numeric).
    predictors : list of str
        List of independent variables (numeric or categorical).
    include_interactions : bool, default False
        Whether to include all pairwise interactions between predictors.
    estimator : function, optional
        Robust estimator from statsmodels (default: HuberT).

    Returns:
    --------
    model : RLMResults
        Fitted robust linear model.
    """
    if not predictors:
        raise ValueError("At least one predictor must be provided.")

    formula_terms = [f"C({v})" if df[v].dtype.name == 'category' or df[v].dtype == object else v for v in predictors]
    sep = " * " if include_interactions else " + "
    formula = f"{outcome} ~ " + sep.join(formula_terms)

    model = smf.rlm(formula=formula, data=df, M=estimator()).fit()
    print(model.summary())
    return model
