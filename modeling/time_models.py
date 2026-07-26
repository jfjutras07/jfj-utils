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

    # Clean data
    train_series = train_series.dropna()
    test_series = test_series.dropna()

    if len(train_series) < 20:
        raise ValueError(
            "train_series must contain at least 20 observations."
        )

    # Fit AutoARIMA
    model = auto_arima(
        train_series,
        seasonal=seasonal,
        m=m,
        information_criterion=information_criterion,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
        max_p=max_p,
        max_q=max_q,
        max_P=max_P,
        max_Q=max_Q
    )

    # Forecast
    forecast = pd.Series(
        model.predict(
            n_periods=len(test_series)
        ),
        index=test_series.index,
        name="Forecast"
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
    forecast_horizon
):
    """
    Generate a drift forecast based on the average historical trend.

    The method assumes that future values will continue following
    the average slope observed between the first and last observations.

    Parameters
    ----------
    train_series : pandas.Series
        Historical time series used for training.

    forecast_horizon : int
        Number of future periods to forecast.

    Returns
    -------
    pandas.Series
        Forecasted values.
    """

    # Validate inputs
    if not isinstance(train_series, pd.Series):
        raise TypeError(
            "train_series must be a pandas Series."
        )

    if train_series.empty:
        raise ValueError(
            "train_series cannot be empty."
        )

    if not isinstance(forecast_horizon, int):
        raise TypeError(
            "forecast_horizon must be an integer."
        )

    if forecast_horizon <= 0:
        raise ValueError(
            "forecast_horizon must be positive."
        )

    # Remove missing observations
    train_series = train_series.dropna()

    if len(train_series) < 2:
        raise ValueError(
            "train_series must contain at least two observations."
        )

    # Calculate drift
    first_value = train_series.iloc[0]
    last_value = train_series.iloc[-1]

    n_periods = len(train_series)

    drift = (
        (last_value - first_value)
        /
        (n_periods - 1)
    )

    forecast_values = [
        last_value + drift * step
        for step in range(1, forecast_horizon + 1)
    ]

    # Preserve datetime index when available
    if isinstance(train_series.index, pd.DatetimeIndex):

        future_index = pd.date_range(
            start=train_series.index[-1],
            periods=forecast_horizon + 1,
            freq=train_series.index.inferred_freq
        )[1:]

    else:
        future_index = range(
            forecast_horizon
        )

    forecast = pd.Series(
        forecast_values,
        index=future_index,
        name="Forecast"
    )

    return forecast

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
    forecast_horizon,
    window_size=3
):
    """
    Generate a moving average forecast.

    The method forecasts future values using the average of the most
    recent observations. It assumes that short-term fluctuations are
    smoothed and future demand follows the recent average level.

    Parameters
    ----------
    train_series : pandas.Series
        Historical time series used for training.

    forecast_horizon : int
        Number of future periods to forecast.

    window_size : int, default=3
        Number of recent observations used to compute the moving average.

    Returns
    -------
    pandas.Series
        Forecasted values.
    """

    # Validate inputs
    if not isinstance(train_series, pd.Series):
        raise TypeError(
            "train_series must be a pandas Series."
        )

    if train_series.empty:
        raise ValueError(
            "train_series cannot be empty."
        )

    if not isinstance(forecast_horizon, int):
        raise TypeError(
            "forecast_horizon must be an integer."
        )

    if forecast_horizon <= 0:
        raise ValueError(
            "forecast_horizon must be positive."
        )

    if not isinstance(window_size, int):
        raise TypeError(
            "window_size must be an integer."
        )

    if window_size <= 0:
        raise ValueError(
            "window_size must be positive."
        )

    # Remove missing observations
    train_series = train_series.dropna()

    if len(train_series) < window_size:
        raise ValueError(
            "train_series must contain at least window_size observations."
        )

    # Calculate recent average
    recent_average = (
        train_series
        .iloc[-window_size:]
        .mean()
    )

    # Preserve datetime index when available
    if isinstance(train_series.index, pd.DatetimeIndex):

        future_index = pd.date_range(
            start=train_series.index[-1],
            periods=forecast_horizon + 1,
            freq=train_series.index.inferred_freq
        )[1:]

    else:
        future_index = range(
            forecast_horizon
        )

    forecast = pd.Series(
        [recent_average] * forecast_horizon,
        index=future_index,
        name="Forecast"
    )

    return forecast

#--- Function : naive_forecast ---
def naive_forecast(
    train_series,
    forecast_horizon
):
    """
    Generate a naive time series forecast using the last observed value.

    The forecast assumes that future values will remain equal
    to the most recent observed value.

    Parameters
    ----------
    train_series : pandas.Series
        Historical time series used for training.

    forecast_horizon : int
        Number of future periods to forecast.

    Returns
    -------
    pandas.Series
        Forecasted values.

    """

    if not isinstance(train_series, pd.Series):
        raise TypeError("train_series must be a pandas Series.")

    if train_series.empty:
        raise ValueError("train_series cannot be empty.")

    if forecast_horizon <= 0:
        raise ValueError("forecast_horizon must be positive.")

    last_value = train_series.iloc[-1]

    forecast = pd.Series(
        np.repeat(last_value, forecast_horizon),
        name="Forecast"
    )

    return forecast

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
    forecast_horizon,
    season_length
):
    """
    Generate a seasonal naive time series forecast.

    The forecast assumes that future values will repeat the values
    observed during the previous seasonal cycle.

    Parameters
    ----------
    train_series : pandas.Series
        Historical time series used for training.

    forecast_horizon : int
        Number of future periods to forecast.

    season_length : int
        Number of observations in one seasonal cycle.
        Examples:
            7  -> weekly seasonality in daily data
            12 -> yearly seasonality in monthly data
            24 -> daily seasonality in hourly data

    Returns
    -------
    pandas.Series
        Forecasted values.
    """

    # Validate inputs
    if not isinstance(train_series, pd.Series):
        raise TypeError(
            "train_series must be a pandas Series."
        )

    if train_series.empty:
        raise ValueError(
            "train_series cannot be empty."
        )

    if not isinstance(forecast_horizon, int):
        raise TypeError(
            "forecast_horizon must be an integer."
        )

    if forecast_horizon <= 0:
        raise ValueError(
            "forecast_horizon must be positive."
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
    train_series = train_series.dropna()

    if len(train_series) < season_length:
        raise ValueError(
            "train_series must contain at least one complete seasonal cycle."
        )

    # Extract last seasonal pattern
    seasonal_pattern = train_series.iloc[-season_length:]

    forecast_values = np.tile(
        seasonal_pattern.values,
        int(np.ceil(forecast_horizon / season_length))
    )[:forecast_horizon]

    # Preserve datetime index when available
    if isinstance(train_series.index, pd.DatetimeIndex):

        future_index = pd.date_range(
            start=train_series.index[-1],
            periods=forecast_horizon + 1,
            freq=train_series.index.inferred_freq
        )[1:]

    else:
        future_index = range(
            forecast_horizon
        )

    forecast = pd.Series(
        forecast_values,
        index=future_index,
        name="Forecast"
    )

    return forecast
