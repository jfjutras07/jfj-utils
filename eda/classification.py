import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

#---Function:logistic_regression---
def logistic_regression(df, outcome, predictors):
    """
    Perform a binary logistic regression for statistical inference.
    Used when the dependent variable is binary (0/1).
    Returns a statsmodels object with p-values and summary.
    """
    if not predictors:
        raise ValueError("At least one predictor must be provided.")

    #Build formula string for statsmodels
    formula = f"{outcome} ~ " + " + ".join(predictors)

    #Fit logistic regression using logit
    model = smf.logit(formula, data=df).fit()
    
    print("--- Logistic Regression Summary ---")
    print(model.summary())
    return model
