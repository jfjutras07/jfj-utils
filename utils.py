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
    Supports passing extra arguments and handles the fit/transform logic
    while preserving or updating column names based on the function's output.
    """
    def __init__(self, func: Callable, **kwargs):
        self.func = func
        self.kwargs = kwargs
        self.feature_names_out_ = None

    def fit(self, X: pd.DataFrame, y=None):
        # Initial capture of feature names
        self.feature_names_out_ = X.columns.tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        result = self.func(X, **self.kwargs)
        
        # Ensure the output is a DataFrame and update feature names if they changed
        if isinstance(result, pd.DataFrame):
            self.feature_names_out_ = result.columns.tolist()
            return result.reset_index(drop=True)
        else:
            # Fallback for numpy arrays or other formats
            return pd.DataFrame(result, columns=self.feature_names_out_).reset_index(drop=True)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_out_)
        
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

