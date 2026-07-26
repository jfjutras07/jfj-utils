import pandas as pd
import numpy as np

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
