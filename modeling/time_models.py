import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from pmdarima import auto_arima

#--- Function : arima_forecast ---
def arima_forecast(
    train_series,
    test_series,
    order=(1, 1, 1)
):
    """
    Generate forecasts using ARIMA.

    ARIMA models temporal dependencies through autoregressive terms,
    differencing, and moving average components.

    Parameters
    ----------
    train_series : pandas.Series
        Historical training time series.

    test_series : pandas.Series
        Observed values used for evaluation.

    order : tuple, default=(1,1,1)
        ARIMA parameters:
            p = autoregressive order
            d = differencing order
            q = moving average order

    Returns
    -------
    dict
        Fitted model, forecast, evaluation metrics,
        and model parameters.
    """

    # Validate inputs
    if not isinstance(train_series, pd.Series):
        raise TypeError(
            "train_series must be a pandas Series."
        )

    if not isinstance(test_series, pd.Series):
        raise TypeError(
            "test_series must be a pandas Series."
        )

    if train_series.empty:
        raise ValueError(
            "train_series cannot be empty."
        )

    if test_series.empty:
        raise ValueError(
            "test_series cannot be empty."
        )

    if (
        not isinstance(order, tuple)
        or len(order) != 3
        or not all(isinstance(x, int) and x >= 0 for x in order)
    ):
        raise ValueError(
            "order must be a tuple of three non-negative integers (p,d,q)."
        )

    # Clean series
    train_series = train_series.dropna()
    test_series = test_series.dropna()

    if len(train_series) < 10:
        raise ValueError(
            "train_series must contain at least 10 observations."
        )

    # Fit ARIMA model
    model = ARIMA(
        train_series,
        order=order
    )

    fitted_model = model.fit()

    # Forecast
    forecast = fitted_model.forecast(
        steps=len(test_series)
    )

    forecast.index = test_series.index

    # Evaluation metrics
    metrics = {
        "MAE": mean_absolute_error(
            test_series,
            forecast
        ),
        "RMSE": np.sqrt(
            mean_squared_error(
                test_series,
                forecast
            )
        )
    }

    return {
        "model": fitted_model,
        "forecast": forecast,
        "metrics": metrics,
        "parameters": {
            "order": order
        }
    }

#--- Function : auto_arima_forecast ---
def auto_arima_forecast(
    train_series,
    test_series,
    seasonal=True,
    m=7,
    information_criterion="aic",
    max_p=5,
    max_q=5,
    max_P=2,
    max_Q=2
):
    """
    Generate forecasts using AutoARIMA.

    Automatically identifies optimal ARIMA/SARIMA parameters
    based on statistical criteria.

    Parameters
    ----------
    train_series : pandas.Series
        Historical training time series.

    test_series : pandas.Series
        Observed values used for evaluation.

    seasonal : bool, default=True
        Enable seasonal ARIMA search.

    m : int, default=7
        Seasonal period length.

    information_criterion : str, default="aic"
        Model selection criterion.

    max_p : int
        Maximum autoregressive order.

    max_q : int
        Maximum moving average order.

    max_P : int
        Maximum seasonal autoregressive order.

    max_Q : int
        Maximum seasonal moving average order.

    Returns
    -------
    dict
        Fitted model, forecast, metrics,
        and selected parameters.
    """

    # Validation
    if not isinstance(train_series, pd.Series):
        raise TypeError(
            "train_series must be a pandas Series."
        )

    if not isinstance(test_series, pd.Series):
        raise TypeError(
            "test_series must be a pandas Series."
        )

    if train_series.empty:
        raise ValueError(
            "train_series cannot be empty."
        )

    if test_series.empty:
        raise ValueError(
            "test_series cannot be empty."
        )

    if m <= 0:
        raise ValueError(
            "m must be a positive integer."
        )

    # Clean data
    train_series = train_series.dropna()
    test_series = test_series.dropna()

    if len(train_series) < 20:
        raise ValueError(
            "train_series must contain at least 20 observations."
        )

    # Seasonal models require enough cycles
    if seasonal and len(train_series) < 2 * m:
        seasonal = False


    # Fit AutoARIMA
    model = auto_arima(
        train_series,
        seasonal=seasonal,
        m=m if seasonal else 1,
        information_criterion=information_criterion,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
        max_p=max_p,
        max_q=max_q,
        max_P=max_P if seasonal else 0,
        max_Q=max_Q if seasonal else 0
    )


    # Forecast
    forecast = pd.Series(
        model.predict(
            n_periods=len(test_series)
        ),
        index=test_series.index,
        name="Forecast"
    )


    # Check invalid forecasts
    if forecast.isna().any():

        raise ValueError(
            "AutoARIMA generated NaN forecasts. "
            "Increase training data or modify seasonal parameters."
        )


    # Metrics
    metrics = {
        "MAE": mean_absolute_error(
            test_series,
            forecast
        ),
        "RMSE": np.sqrt(
            mean_squared_error(
                test_series,
                forecast
            )
        )
    }


    return {
        "model": model,
        "forecast": forecast,
        "metrics": metrics,
        "parameters": {
            "order": model.order,
            "seasonal_order": model.seasonal_order
        }
    }
    
#--- Function : drift_forecast ---
def drift_forecast(
    train_series,
    test_series
):
    """
    Generate forecasts using the drift method.

    The drift method extrapolates the average historical trend
    observed between the first and last observations.

    Parameters
    ----------
    train_series : pandas.Series
        Historical training time series.

    test_series : pandas.Series
        Observed values used for evaluation.

    Returns
    -------
    dict
        Forecast, evaluation metrics,
        and model parameters.
    """

    # Validate inputs
    if not isinstance(train_series, pd.Series):
        raise TypeError(
            "train_series must be a pandas Series."
        )

    if not isinstance(test_series, pd.Series):
        raise TypeError(
            "test_series must be a pandas Series."
        )

    if train_series.empty:
        raise ValueError(
            "train_series cannot be empty."
        )

    if test_series.empty:
        raise ValueError(
            "test_series cannot be empty."
        )

    # Clean series
    train_series = (
        train_series
        .dropna()
        .sort_index()
    )

    test_series = (
        test_series
        .dropna()
        .sort_index()
    )

    if len(train_series) < 2:
        raise ValueError(
            "train_series must contain at least two observations."
        )

    # Compute drift
    first_value = train_series.iloc[0]
    last_value = train_series.iloc[-1]

    drift = (
        (last_value - first_value)
        /
        (len(train_series) - 1)
    )

    forecast_values = [
        last_value + drift * step
        for step in range(
            1,
            len(test_series) + 1
        )
    ]

    # Forecast
    forecast = pd.Series(
        forecast_values,
        index=test_series.index,
        name="Forecast"
    )

    # Evaluation metrics
    metrics = {
        "MAE": mean_absolute_error(
            test_series,
            forecast
        ),
        "RMSE": np.sqrt(
            mean_squared_error(
                test_series,
                forecast
            )
        )
    }

    return {
        "model": None,
        "forecast": forecast,
        "metrics": metrics,
        "parameters": {
            "method": "drift"
        }
    }

#--- Function : exponential_smoothing_forecast ---
def exponential_smoothing_forecast(
    train_series,
    test_series,
    trend="add",
    seasonal=None,
    seasonal_periods=None
):
    """
    Generate forecasts using Exponential Smoothing (Holt-Winters).

    The method models level, trend, and optional seasonal patterns.
    It is suitable for time series with evolving patterns and
    recurring seasonal effects.

    Parameters
    ----------
    train_series : pandas.Series
        Historical training time series.

    test_series : pandas.Series
        Observed values used for evaluation.

    trend : {"add", "mul", None}, default="add"
        Type of trend component.

    seasonal : {"add", "mul", None}, default=None
        Type of seasonal component.

    seasonal_periods : int, optional
        Number of observations in one seasonal cycle.

    Returns
    -------
    dict
        Fitted model, forecast, and evaluation metrics.
    """

    # Validate inputs
    if not isinstance(train_series, pd.Series):
        raise TypeError(
            "train_series must be a pandas Series."
        )

    if not isinstance(test_series, pd.Series):
        raise TypeError(
            "test_series must be a pandas Series."
        )

    if train_series.empty:
        raise ValueError(
            "train_series cannot be empty."
        )

    if test_series.empty:
        raise ValueError(
            "test_series cannot be empty."
        )

    if trend not in ["add", "mul", None]:
        raise ValueError(
            "trend must be 'add', 'mul', or None."
        )

    if seasonal not in ["add", "mul", None]:
        raise ValueError(
            "seasonal must be 'add', 'mul', or None."
        )

    if seasonal is not None and seasonal_periods is None:
        raise ValueError(
            "seasonal_periods must be provided when seasonal component is used."
        )

    # Clean series
    train_series = train_series.dropna()
    test_series = test_series.dropna()

    if len(train_series) < 2:
        raise ValueError(
            "train_series must contain at least two observations."
        )

    # Fit model
    model = ExponentialSmoothing(
        train_series,
        trend=trend,
        seasonal=seasonal,
        seasonal_periods=seasonal_periods
    )

    fitted_model = model.fit()

    # Forecast
    forecast = fitted_model.forecast(
        len(test_series)
    )

    forecast.index = test_series.index

    # Evaluation metrics
    metrics = {
        "MAE": mean_absolute_error(
            test_series,
            forecast
        ),
        "RMSE": np.sqrt(
            mean_squared_error(
                test_series,
                forecast
            )
        )
    }

    return {
        "model": fitted_model,
        "forecast": forecast,
        "metrics": metrics,
        "parameters": {
            "trend": trend,
            "seasonal": seasonal,
            "seasonal_periods": seasonal_periods
        }
    }

#--- Function : moving_average_forecast ---
def moving_average_forecast(
    train_series,
    test_series,
    window_size=3
):
    """
    Generate forecasts using the moving average method.

    The moving average method assumes that future observations
    will remain close to the average of the most recent values.

    Parameters
    ----------
    train_series : pandas.Series
        Historical training time series.

    test_series : pandas.Series
        Observed values used for evaluation.

    window_size : int, default=3
        Number of recent observations used to compute
        the moving average.

    Returns
    -------
    dict
        Forecast, evaluation metrics,
        and model parameters.
    """

    # Validate inputs
    if not isinstance(train_series, pd.Series):
        raise TypeError(
            "train_series must be a pandas Series."
        )

    if not isinstance(test_series, pd.Series):
        raise TypeError(
            "test_series must be a pandas Series."
        )

    if train_series.empty:
        raise ValueError(
            "train_series cannot be empty."
        )

    if test_series.empty:
        raise ValueError(
            "test_series cannot be empty."
        )

    if not isinstance(window_size, int):
        raise TypeError(
            "window_size must be an integer."
        )

    if window_size <= 0:
        raise ValueError(
            "window_size must be positive."
        )

    # Clean series
    train_series = (
        train_series
        .dropna()
        .sort_index()
    )

    test_series = (
        test_series
        .dropna()
        .sort_index()
    )

    if len(train_series) < window_size:
        raise ValueError(
            "train_series must contain at least window_size observations."
        )

    # Compute moving average
    moving_average = (
        train_series
        .iloc[-window_size:]
        .mean()
    )

    # Forecast
    forecast = pd.Series(
        np.repeat(
            moving_average,
            len(test_series)
        ),
        index=test_series.index,
        name="Forecast"
    )

    # Evaluation metrics
    metrics = {
        "MAE": mean_absolute_error(
            test_series,
            forecast
        ),
        "RMSE": np.sqrt(
            mean_squared_error(
                test_series,
                forecast
            )
        )
    }

    return {
        "model": None,
        "forecast": forecast,
        "metrics": metrics,
        "parameters": {
            "window_size": window_size
        }
    }

#--- Function : naive_forecast ---
def naive_forecast(
    train_series,
    test_series
):
    """
    Generate forecasts using the naive forecasting method.

    The naive method assumes that future observations
    will remain equal to the last observed value.

    Parameters
    ----------
    train_series : pandas.Series
        Historical training time series.

    test_series : pandas.Series
        Observed values used for evaluation.

    Returns
    -------
    dict
        Forecast, evaluation metrics,
        and model parameters.
    """

    # Validate inputs
    if not isinstance(train_series, pd.Series):
        raise TypeError(
            "train_series must be a pandas Series."
        )

    if not isinstance(test_series, pd.Series):
        raise TypeError(
            "test_series must be a pandas Series."
        )

    if train_series.empty:
        raise ValueError(
            "train_series cannot be empty."
        )

    if test_series.empty:
        raise ValueError(
            "test_series cannot be empty."
        )

    # Clean series
    train_series = (
        train_series
        .dropna()
        .sort_index()
    )

    test_series = (
        test_series
        .dropna()
        .sort_index()
    )

    if len(train_series) < 2:
        raise ValueError(
            "train_series must contain at least 2 observations."
        )

    # Generate forecast
    forecast = pd.Series(
        np.repeat(
            train_series.iloc[-1],
            len(test_series)
        ),
        index=test_series.index,
        name="Forecast"
    )

    # Evaluation metrics
    metrics = {
        "MAE": mean_absolute_error(
            test_series,
            forecast
        ),
        "RMSE": np.sqrt(
            mean_squared_error(
                test_series,
                forecast
            )
        )
    }

    return {
        "model": None,
        "forecast": forecast,
        "metrics": metrics,
        "parameters": {
            "method": "naive"
        }
    }
    
#--- Function : prophet_forecast ---
def prophet_forecast(
    train_series,
    test_series,
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False
):
    """
    Generate forecasts using Prophet.

    Prophet models time series using trend components,
    multiple seasonalities, and calendar effects.

    Parameters
    ----------
    train_series : pandas.Series
        Historical training time series.
        Index must be a DatetimeIndex.

    test_series : pandas.Series
        Observed values used for evaluation.
        Index must be a DatetimeIndex.

    yearly_seasonality : bool, default=True
        Enable yearly seasonal patterns.

    weekly_seasonality : bool, default=True
        Enable weekly seasonal patterns.

    daily_seasonality : bool, default=False
        Enable daily seasonal patterns.

    Returns
    -------
    dict
        Fitted Prophet model, forecast,
        evaluation metrics, and parameters.
    """

    # Validate inputs
    if not isinstance(train_series, pd.Series):
        raise TypeError(
            "train_series must be a pandas Series."
        )

    if not isinstance(test_series, pd.Series):
        raise TypeError(
            "test_series must be a pandas Series."
        )

    if train_series.empty:
        raise ValueError(
            "train_series cannot be empty."
        )

    if test_series.empty:
        raise ValueError(
            "test_series cannot be empty."
        )

    if not isinstance(
        train_series.index,
        pd.DatetimeIndex
    ):
        raise TypeError(
            "train_series index must be a DatetimeIndex."
        )

    if not isinstance(
        test_series.index,
        pd.DatetimeIndex
    ):
        raise TypeError(
            "test_series index must be a DatetimeIndex."
        )

    # Clean series
    train_series = (
        train_series
        .dropna()
        .sort_index()
    )

    test_series = (
        test_series
        .dropna()
        .sort_index()
    )

    if len(train_series) < 10:
        raise ValueError(
            "train_series must contain at least 10 observations."
        )

    # Prepare Prophet format
    train_df = (
        train_series
        .rename("y")
        .reset_index()
        .rename(
            columns={
                train_series.index.name or "index": "ds"
            }
        )
    )

    # Fit Prophet model
    model = Prophet(
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=weekly_seasonality,
        daily_seasonality=daily_seasonality
    )

    model.fit(train_df)

    # Forecast future dates
    future = pd.DataFrame(
        {
            "ds": test_series.index
        }
    )

    prediction = model.predict(
        future
    )

    forecast = pd.Series(
        prediction["yhat"].values,
        index=test_series.index,
        name="Forecast"
    )

    # Evaluation metrics
    metrics = {
        "MAE": mean_absolute_error(
            test_series,
            forecast
        ),
        "RMSE": np.sqrt(
            mean_squared_error(
                test_series,
                forecast
            )
        )
    }

    return {
        "model": model,
        "forecast": forecast,
        "metrics": metrics,
        "parameters": {
            "yearly_seasonality": yearly_seasonality,
            "weekly_seasonality": weekly_seasonality,
            "daily_seasonality": daily_seasonality
        }
    }

#--- Function : sarima_forecast ---
def sarima_forecast(
    train_series,
    test_series,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 7)
):
    """
    Generate forecasts using SARIMA.

    SARIMA extends ARIMA by adding seasonal autoregressive,
    differencing, and moving average components.

    Parameters
    ----------
    train_series : pandas.Series
        Historical training time series.

    test_series : pandas.Series
        Observed values used for evaluation.

    order : tuple, default=(1,1,1)
        Non-seasonal ARIMA parameters:
            (p, d, q)

    seasonal_order : tuple, default=(1,1,1,7)
        Seasonal parameters:
            (P, D, Q, s)

    Returns
    -------
    dict
        Fitted model, forecast, evaluation metrics,
        and model parameters.
    """

    # Validate inputs
    if not isinstance(train_series, pd.Series):
        raise TypeError(
            "train_series must be a pandas Series."
        )

    if not isinstance(test_series, pd.Series):
        raise TypeError(
            "test_series must be a pandas Series."
        )

    if train_series.empty:
        raise ValueError(
            "train_series cannot be empty."
        )

    if test_series.empty:
        raise ValueError(
            "test_series cannot be empty."
        )

    if (
        not isinstance(order, tuple)
        or len(order) != 3
        or not all(isinstance(x, int) and x >= 0 for x in order)
    ):
        raise ValueError(
            "order must be a tuple of three non-negative integers (p,d,q)."
        )

    if (
        not isinstance(seasonal_order, tuple)
        or len(seasonal_order) != 4
        or not all(
            isinstance(x, int) and x >= 0
            for x in seasonal_order
        )
    ):
        raise ValueError(
            "seasonal_order must be a tuple of four non-negative integers (P,D,Q,s)."
        )

    # Clean series
    train_series = train_series.dropna()
    test_series = test_series.dropna()

    if len(train_series) < 10:
        raise ValueError(
            "train_series must contain at least 10 observations."
        )

    # Fit SARIMA model
    model = SARIMAX(
        train_series,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    fitted_model = model.fit(
        disp=False
    )

    # Forecast
    forecast = fitted_model.forecast(
        steps=len(test_series)
    )

    forecast.index = test_series.index

    # Evaluation metrics
    metrics = {
        "MAE": mean_absolute_error(
            test_series,
            forecast
        ),
        "RMSE": np.sqrt(
            mean_squared_error(
                test_series,
                forecast
            )
        )
    }

    return {
        "model": fitted_model,
        "forecast": forecast,
        "metrics": metrics,
        "parameters": {
            "order": order,
            "seasonal_order": seasonal_order
        }
    }

#--- Function : seasonal_naive_forecast ---
def seasonal_naive_forecast(
    train_series,
    test_series,
    season_length
):
    """
    Generate forecasts using the seasonal naive method.

    The seasonal naive method assumes that future observations
    will repeat the values observed during the previous
    seasonal cycle.

    Parameters
    ----------
    train_series : pandas.Series
        Historical training time series.

    test_series : pandas.Series
        Observed values used for evaluation.

    season_length : int
        Number of observations in one seasonal cycle.

    Returns
    -------
    dict
        Forecast, evaluation metrics,
        and model parameters.
    """

    # Validate inputs
    if not isinstance(train_series, pd.Series):
        raise TypeError(
            "train_series must be a pandas Series."
        )

    if not isinstance(test_series, pd.Series):
        raise TypeError(
            "test_series must be a pandas Series."
        )

    if train_series.empty:
        raise ValueError(
            "train_series cannot be empty."
        )

    if test_series.empty:
        raise ValueError(
            "test_series cannot be empty."
        )

    if not isinstance(season_length, int):
        raise TypeError(
            "season_length must be an integer."
        )

    if season_length <= 0:
        raise ValueError(
            "season_length must be positive."
        )

    # Clean series
    train_series = (
        train_series
        .dropna()
        .sort_index()
    )

    test_series = (
        test_series
        .dropna()
        .sort_index()
    )

    if len(train_series) < season_length:
        raise ValueError(
            "train_series must contain at least one complete seasonal cycle."
        )

    # Extract seasonal pattern
    seasonal_pattern = (
        train_series
        .iloc[-season_length:]
        .values
    )

    forecast_values = np.tile(
        seasonal_pattern,
        int(
            np.ceil(
                len(test_series)
                /
                season_length
            )
        )
    )[:len(test_series)]

    # Forecast
    forecast = pd.Series(
        forecast_values,
        index=test_series.index,
        name="Forecast"
    )

    # Evaluation metrics
    metrics = {
        "MAE": mean_absolute_error(
            test_series,
            forecast
        ),
        "RMSE": np.sqrt(
            mean_squared_error(
                test_series,
                forecast
            )
        )
    }

    return {
        "model": None,
        "forecast": forecast,
        "metrics": metrics,
        "parameters": {
            "season_length": season_length
        }
    }
