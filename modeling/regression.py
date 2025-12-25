import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

#--- Function : robust_regression ---
def robust_regression(df, outcome, factor, covariates, estimator=sm.robust.norms.HuberT):
    """
    Perform a robust ANCOVA-like regression using RLM (Robust Linear Model).
    
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
