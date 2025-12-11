import time
import datetime
from functools import wraps

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

