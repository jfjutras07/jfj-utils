import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Optional

#--- Class : column_dropper ---
class column_dropper(BaseEstimator, TransformerMixin):
    """
    Drops specified columns and can automatically remove constant columns 
    (zero variance) detected during the fit process.
    """
    def __init__(self, columns: Optional[List[str]] = None, drop_constant: bool = False):
        self.columns = columns if columns else []
        self.drop_first = drop_constant
        self.constant_cols_ = []

    def fit(self, X: pd.DataFrame, y=None):
        # Automatically identify columns with only 1 unique value
        if self.drop_first:
            self.constant_cols_ = [col for col in X.columns if X[col].nunique() <= 1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        # Combine manual list and detected constant columns
        total_to_drop = list(set(self.columns + self.constant_cols_))
        
        # Drop only if columns exist in the current dataframe
        existing_cols = [c for c in total_to_drop if c in X.columns]
        return X.drop(columns=existing_cols)
