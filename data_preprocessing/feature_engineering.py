import pandas as pd
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Dict, List

#--- Class : ratio_generator ---
class ratio_generator(BaseEstimator, TransformerMixin):
    """
    Generates ratio features between pairs of columns with explicit names.
    Input format: {'NewFeaturePrefix': ['NumeratorCol', 'DenominatorCol']}
    Example output column: 'Tenure_Ratio_YearsAtCompany_TotalWorkingYears'
    """
    def __init__(self, ratio_mappings: Dict[str, List[str]], fill_na_value: float = 0.0):
        self.ratio_mappings = ratio_mappings
        self.fill_na_value = fill_na_value
        self.feature_names_ = []

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        self.feature_names_ = []
        
        for prefix, (num_col, den_col) in self.ratio_mappings.items():
            ratio_name = f"{prefix}_Ratio_{num_col}_{den_col}"
            # Safe division
            den_safe = X[den_col].replace(0, np.nan)
            X[ratio_name] = X[num_col] / den_safe
            if self.fill_na_value is not None:
                X[ratio_name] = X[ratio_name].fillna(self.fill_na_value)
            self.feature_names_.append(ratio_name)
        
        return X

# --- Function : mi_classification ---
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
