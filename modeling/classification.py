
# --- Function : logistic_regression ---
def logistic_regression(df, outcome, predictors):
    """
    Perform a binary logistic regression.

    When to use:
    - Dependent variable is binary (0/1).
    - Predictors can be continuous or categorical.
    
    Example:
    --------
    #Fit a logistic regression of y (0/1) on x1 and x2
    data = pd.DataFrame({
        'x1': [1, 2, 3, 4, 5],
        'x2': [2, 3, 2, 5, 4],
        'y': [0, 0, 1, 1, 1]
    })
    model = logistic_regression(data, outcome='y', predictors=['x1', 'x2'])

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset containing predictors and outcome.
    outcome : str
        Dependent binary variable (0/1).
    predictors : list of str
        List of independent numeric variables.

    Returns:
    --------
    model : statsmodels RegressionResults
        Fitted logistic regression model.
    """
    if not predictors:
        raise ValueError("At least one predictor must be provided.")

    # Build formula string
    formula = f"{outcome} ~ " + " + ".join(predictors)

    # Fit logistic regression
    model = smf.logit(formula, data=df).fit()
    print(model.summary())

    return model
