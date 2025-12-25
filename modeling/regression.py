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

    When to use:
    - Dependent variable is positive and continuous.
    - Data are right-skewed.
    - Common in time, cost, concentrations.

    Example:
    --------
    data = pd.DataFrame({'x':[1,2,3,4,5],
                         'y':[2.1,3.5,6.0,10.2,15.1]})
    model = gamma_regression(data, outcome='y', predictors=['x'])

    Parameters:
    -----------
    df : pd.DataFrame
    outcome : str
    predictors : list of str
    link : str, default 'log'

    Returns:
    --------
    model : statsmodels GLMResults
    """
    formula = f"{outcome} ~ " + " + ".join(predictors)
    
    link_func = sm.families.links.log() if link=='log' else sm.families.links.identity()
    
    model = smf.glm(formula=formula, data=df, family=sm.families.Gamma(link=link_func)).fit()
    
    print(model.summary())
    
    return model

# --- Function : linear_regression ---
def linear_regression(df, outcome, predictors):
    """
    Perform a multiple linear regression.

    When to use:
    - Dependent variable is continuous and approximately normal.
    - Predictor can be continuous or categorical.

    Example:
    --------
    #Fit a regression of y on x1 and x2
    data = pd.DataFrame({
        'x1': [1, 2, 3, 4, 5],
        'x2': [2, 3, 2, 5, 4],
        'y': [2.1, 4.5, 5.9, 8.2, 10.1]
    })
    model = linear_regression(data, outcome='y', predictors=['x1', 'x2'])

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
    
    #Build formula string
    formula = f"{outcome} ~ " + " + ".join(predictors)
    
    #Fit linear model
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

    When to use:
    - Dependent variable is continuous.
    - Relationship with predictor is non-linear but can be approximated by polynomials.
    
    Example:
    --------
    #Fit a cubic regression of y on x
    data = pd.DataFrame({
        'x': [1, 2, 3, 4, 5],
        'y': [2.1, 4.5, 9.0, 16.2, 25.1]
    })
    model = polynomial_regression(data, outcome='y', predictor='x', max_degree=3)

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
    
    #Create polynomial terms for degrees 2..max_degree
    poly_terms = []
    for d in range(2, max_degree + 1):
        col_name = f"{predictor}^{d}"
        df_poly[col_name] = df_poly[predictor]**d
        poly_terms.append(col_name)
    
    #Build formula string
    formula = f"{outcome} ~ {predictor}"
    if poly_terms:
        formula += " + " + " + ".join(poly_terms)
    
    #Fit linear model
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

#--- Function : robust_regression ---
def robust_regression(df, outcome, factor, covariates, estimator=sm.robust.norms.HuberT):
    """
    Perform a robust ANCOVA-like regression using RLM (Robust Linear Model).

    When to use:
    - Dependent variable is continuous.
    - Outliers may distort standard OLS estimates.
    - Similar to ANCOVA when factor is categorical and covariates numeric.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset containing outcome, factor, and covariates.
    outcome : str
        Name of the dependent variable (numeric).
    factor : str
        Name of the categorical independent variable (factor).
    covariates : list of str
        List of covariate column names (numeric).
    estimator : function, optional
        Robust estimator function from statsmodels (default: HuberT).
    
    Returns:
    --------
    fitted_model : RLMResults
        Fitted robust linear model object from statsmodels.
    """
    #Ensure factor is categorical
    df[factor] = df[factor].astype('category')
    
    #Build formula string
    formula = f"{outcome} ~ {factor}"
    if covariates:
        formula += " + " + " + ".join(covariates)
    
    #Fit robust linear model
    model = smf.rlm(formula=formula, data=df, M=estimator()).fit()
    
    #Print summary
    print(model.summary())
    
    return model
