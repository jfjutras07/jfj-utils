import time
import datetime
from functools import wraps
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Callable, Dict, Any, Optional

#--- Class : generic_transformer ---
class generic_transformer(BaseEstimator, TransformerMixin):
    """
    Adapter to use any custom function inside a Scikit-Learn Pipeline.
    Supports passing extra arguments and handles the fit/transform logic.
    Preserves column names.
    """
    def __init__(self, func: Callable, **kwargs):
        self.func = func
        self.kwargs = kwargs
        self.feature_names_in_ = None

    def fit(self, X: pd.DataFrame, y=None):
        # Store column names to ensure consistency
        self.feature_names_in_ = X.columns.tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        result = self.func(X, **self.kwargs)
        
        # Ensure output is a DataFrame with correct column names
        if isinstance(result, pd.DataFrame):
            return result
        else:
            return pd.DataFrame(result, columns=self.feature_names_in_)

# --- Function : log_message ---
def log_message(message: str, level: str = "INFO") -> None:
    """
    Simple centralized logging function.

    Parameters
    ----------
    message : str
        Text message to log.
    level : str, default "INFO"
        Log level (e.g., INFO, WARNING, ERROR).

    Notes
    -----
    Outputs a timestamped message, formatted consistently for auditing.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")


# --- Function : timeit ---
def timeit(func):
    """
    Decorator to measure execution time of a function.

    Parameters
    ----------
    func : callable
        Function to be wrapped.

    Returns
    -------
    callable
        Wrapped function that logs execution time.

    Notes
    -----
    Useful for auditing and performance monitoring of ETL steps.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        log_message(f"Starting '{func.__name__}'...", level="INFO")

        result = func(*args, **kwargs)

        end_time = time.perf_counter()
        elapsed = end_time - start_time
        log_message(
            f"Finished '{func.__name__}' in {elapsed:.4f} seconds.",
            level="INFO"
        )

        return result

    return wrapper

