import pandas as pd
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Dict, List

#--- Class : ratio_generator ---
class ratio_generator(BaseEstimator, TransformerMixin):
    """
    Generates ratio features and handles division by zero.
    """
    def __init__(self, ratio_mappings: Dict[str, List[str]], fill_na_value: float = 0.0, drop_source: bool = True):
        self.ratio_mappings = ratio_mappings
        self.fill_na_value = fill_na_value
        self.drop_source = drop_source
        self.feature_names_ = []

    def fit(self, X: pd.DataFrame, y=None):
        # Define output features during fit
        self.feature_names_ = list(self.ratio_mappings.keys())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        used_cols = set()

        for prefix, (num_col, den_col) in self.ratio_mappings.items():
            # Ensure columns exist
            if num_col not in X.columns or den_col not in X.columns:
                continue
                
            # Replace 0 by NaN to avoid Inf values
            den_safe = X[den_col].replace(0, np.nan)
            X[prefix] = X[num_col] / den_safe

            if self.fill_na_value is not None:
                X[prefix] = X[prefix].fillna(self.fill_na_value)

            used_cols.update([num_col, den_col])

        if self.drop_source:
            X = X.drop(columns=list(used_cols), errors='ignore')

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
