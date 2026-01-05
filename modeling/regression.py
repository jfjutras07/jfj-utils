import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import lifelines
from lifelines import CoxPHFitter

# --- Function : cox_regression ---
def cox_regression(df, duration_col, event_col, covariates):
    """
    Perform Cox Proportional Hazards regression for survival/time-to-event data.

    When to use:
    - Outcome is time-to-event (duration) and event indicator (1=event, 0=censored).
    - Estimate hazard ratios for predictors (covariates).
    - Can handle continuous or categorical covariates.
    - Useful in medical studies, reliability analysis, or any survival analysis.

    Example:
    --------
    # Example dataset
    data = pd.DataFrame({
        'time': [5, 6, 6, 2, 4, 3, 7, 8],
        'event': [1, 0, 1, 1, 0, 1, 0, 1],
        'age': [45, 50, 38, 60, 55, 47, 52, 49],
        'treatment': [0, 1, 0, 1, 0, 1, 1, 0]
    })
    model = cox_regression(data, duration_col='time', event_col='event', covariates=['age', 'treatment'])

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

#---Function: gamma_regression---
def gamma_regression(df, outcome, predictors, link='log'):
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
    link : str, default 'log'
        Link function ('log' or 'identity').

    Returns:
    --------
    model : statsmodels GLMResults
        Fitted Gamma regression model.
    """
    if not predictors:
        raise ValueError("At least one predictor must be provided.")
    
    formula = f"{outcome} ~ " + " * ".join(predictors)
    link_func = sm.families.links.log() if link=='log' else sm.families.links.identity()
    model = smf.glm(formula=formula, data=df, family=sm.families.Gamma(link=link_func)).fit()
    print(model.summary())
    return model

# --- Function: linear_mixed_model ---
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

    #Build formula
    if include_interactions:
        fixed_formula = " * ".join(fixed_effects)  # includes main effects + all pairwise interactions
    else:
        fixed_formula = " + ".join(fixed_effects)  # only main effects

    formula = f"{outcome} ~ {fixed_formula}"
    
    #Fit the model
    model = smf.mixedlm(formula=formula, data=df, groups=df[random_effect])
    model_fit = model.fit(reml=True)
    
    #Print summary
    print(model_fit.summary())
    
    return model_fit

#---Function: linear_regression---
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

    #Build formula
    if include_interactions:
        formula = f"{outcome} ~ " + " * ".join(predictors)  # main effects + interactions
    else:
        formula = f"{outcome} ~ " + " + ".join(predictors)  # main effects only

    #Fit model
    model = smf.ols(formula, data=df).fit()

    #Print summary
    print(model.summary())
    return model

#---Function: poisson_regression---
def poisson_regression(df, outcome, predictors):
    """
    Perform a Poisson regression for count data.

    When to use:
    - Dependent variable is count (0,1,2,...).
    - Counts are not overdispersed (variance ≈ mean).

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset containing predictors and outcome.
    outcome : str
        Dependent variable representing counts (non-negative integers).
    predictors : list of str
        List of independent numeric variables.

    Returns:
    --------
    model : statsmodels RegressionResults
        Fitted Poisson regression model.
    """
    if not predictors:
        raise ValueError("At least one predictor must be provided.")

    formula = f"{outcome} ~ " + " * ".join(predictors)
    model = smf.poisson(formula, data=df).fit()
    print(model.summary())
    return model

# --- Function : polynomial_regression ---
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
    
    df_poly = df.copy()
    poly_terms = []
    for d in range(2, max_degree + 1):
        col_name = f"{predictor}^{d}"
        df_poly[col_name] = df_poly[predictor]**d
        poly_terms.append(col_name)
    
    formula = f"{outcome} ~ {predictor}"
    if poly_terms:
        formula += " + " + " + ".join(poly_terms)
    
    model = smf.ols(formula, data=df_poly).fit()
    print(model.summary())
    return model 

# --- Function : quantile_regression ---
def quantile_regression(df, outcome, predictor, quantile=0.5):
    """
    Perform quantile regression (median or other quantiles).

    When to use:
    - Dependent variable is continuous.
    - Interested in conditional median or other quantiles.
    - Robust to outliers and heteroscedasticity.

    Example:
    --------
    data = pd.DataFrame({'x':[1,2,3,4,5],
                         'y':[2.1,4.5,9.0,16.2,25.1]})
    model = quantile_regression(data, outcome='y', predictor='x', quantile=0.5)

    Parameters:
    -----------
    df : pd.DataFrame
    outcome : str
    predictor : str
    quantile : float (0 < quantile < 1)

    Returns:
    --------
    model : statsmodels RegressionResults
    """
    formula = f"{outcome} ~ {predictor}"
    model = smf.quantreg(formula=formula, data=df).fit(q=quantile)
    print(model.summary())
    return model

#---Function: robust_regression---
def robust_regression(df, outcome, factor=None, covariates=None, predictors=None, estimator=sm.robust.norms.HuberT):
    """
    Perform a robust linear regression using RLM (Robust Linear Model).

    Flexible usage:
    1. ANCOVA-style: specify 'factor' and 'covariates'
    2. Simple regression: specify 'predictors' only

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset containing outcome and predictors.
    outcome : str
        Dependent variable (numeric).
    factor : str, optional
        Categorical independent variable for ANCOVA-style regression.
    covariates : list of str, optional
        List of numeric covariates for ANCOVA-style regression.
    predictors : list of str, optional
        List of independent variables (numeric or categorical) for simple regression.
    estimator : function, optional
        Robust estimator from statsmodels (default: HuberT).

    Returns:
    --------
    model : RLMResults
        Fitted robust linear model.
    """
    # Build formula
    if factor is not None:
        df[factor] = df[factor].astype('category')
        if covariates:
            formula = f"{outcome} ~ C({factor}) * (" + " + ".join(covariates) + ")"
        else:
            formula = f"{outcome} ~ C({factor})"
    elif predictors is not None:
        formula_terms = []
        for var in predictors:
            if df[var].dtype.name == 'category' or df[var].dtype == object:
                formula_terms.append(f"C({var})")
            else:
                formula_terms.append(var)
        formula = f"{outcome} ~ " + " + ".join(formula_terms)
    else:
        raise ValueError("You must provide either 'factor' or 'predictors'.")

    # Fit robust linear model
    model = smf.rlm(formula=formula, data=df, M=estimator()).fit()
    print(model.summary())
    return model
