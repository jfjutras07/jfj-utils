import pandas as pd
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif

# --- Function : mi_regression ---
def mi_regression(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """
    Compute Mutual Information (MI) scores for a regression problem.

    Parameters:
        X : pd.DataFrame
            Feature matrix (can contain categorical and continuous variables)
        y : pd.Series
            Continuous target variable

    Returns:
        pd.Series
            MI scores for each feature, sorted descending
    """
    X = X.copy()
    
    #Factorize categorical columns
    for col in X.select_dtypes(["object", "category"]):
        X[col], _ = X[col].factorize()
    
    #Identify discrete features (integer dtype)
    discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]
    
    #Compute MI scores
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features, random_state=0)
    
    return pd.Series(mi_scores, index=X.columns, name="MI Scores").sort_values(ascending=False)


# --- Funciton : mi_classification ---
def mi_classification(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """
    Compute Mutual Information (MI) scores for a classification problem.

    Parameters:
        X : pd.DataFrame
            Feature matrix (can contain categorical and continuous variables)
        y : pd.Series
            Categorical target variable

    Returns:
        pd.Series
            MI scores for each feature, sorted descending
    """
    X = X.copy()
    
    #Factorize categorical columns
    for col in X.select_dtypes(["object", "category"]):
        X[col], _ = X[col].factorize()
    
    #Identify discrete features (integer dtype)
    discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]
    
    #Compute MI scores
    mi_scores = mutual_info_classif(X, y, discrete_features=discrete_features, random_state=0)
    
    return pd.Series(mi_scores, index=X.columns, name="MI Scores").sort_values(ascending=False)

