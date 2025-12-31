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
    from lifelines import CoxPHFitter

    if not covariates:
        raise ValueError("At least one covariate must be provided.")

    cph = CoxPHFitter()
    cph.fit(df[[duration_col, event_col] + covariates], duration_col=duration_col, event_col=event_col)
    cph.print_summary()
    return cph

# --- Function : gamma_regression ---
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
    
    formula = f"{outcome} ~ " + " + ".join(predictors)
    link_func = sm.families.links.log() if link=='log' else sm.families.links.identity()
    model = smf.glm(formula=formula, data=df, family=sm.families.Gamma(link=link_func)).fit()
    print(model.summary())
    return model 

# --- Function : linear_regression ---
def linear_regression(df, outcome, predictors):
    """
    Perform a multiple linear regression.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset containing predictors and outcome.
    outcome : str
        Dependent variable.
    predictors : list of str
        List of independent numeric variables.

    Returns:
    --------
    model : statsmodels RegressionResults
        Fitted linear regression model.
    """
    if not predictors:
        raise ValueError("At least one predictor must be provided.")
    
    formula = f"{outcome} ~ " + " + ".join(predictors)
    model = smf.ols(formula, data=df).fit()
    print(model.summary())
    return model 

# --- Function : poisson_regression ---
def poisson_regression(df, outcome, predictors):
    """
    Perform a Poisson regression for count data.

    When to use:
    - Dependent variable is count (0,1,2,...).
    - Counts are not overdispersed (variance ≈ mean).

    Example:
    --------
    #Fit a Poisson regression of counts y on x1 and x2
    data = pd.DataFrame({
        'x1': [1, 2, 3, 4, 5],
        'x2': [2, 3, 2, 5, 4],
        'y': [0, 1, 3, 4, 5]
    })
    model = poisson_regression(data, outcome='y', predictors=['x1', 'x2'])

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

    #Build formula string
    formula = f"{outcome} ~ " + " + ".join(predictors)

    #Fit Poisson regression
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

# --- Function : robust_regression ---
def robust_regression(df, outcome, factor, covariates, estimator=sm.robust.norms.HuberT):
    """
    Perform a robust ANCOVA-like regression using RLM (Robust Linear Model).

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset containing outcome, factor, and covariates.
    outcome : str
        Dependent variable (numeric).
    factor : str
        Categorical independent variable.
    covariates : list of str
        List of numeric covariates.
    estimator : function, optional
        Robust estimator from statsmodels (default: HuberT).

    Returns:
    --------
    model : RLMResults
        Fitted robust linear model.
    """
    df[factor] = df[factor].astype('category')
    formula = f"{outcome} ~ {factor}"
    if covariates:
        formula += " + " + " + ".join(covariates)
    
    model = smf.rlm(formula=formula, data=df, M=estimator()).fit()
    print(model.summary())
    return model  

