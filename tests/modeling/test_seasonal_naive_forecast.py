import pandas as pd
from modeling.time_series import seasonal_naive_forecast

#--- Test : seasonal_naive_forecast ---
def test_seasonal_naive_forecast():

    train_series = pd.Series(
        [10, 12, 14, 16, 20, 22, 24],
        name="Demand"
    )

    forecast = seasonal_naive_forecast(
        train_series,
        forecast_horizon=5,
        season_length=3
    )

    # Check output type
    assert isinstance(
        forecast,
        pd.Series
    )

    # Check forecast length
    assert len(forecast) == 5

    # Check seasonal repetition logic
    expected = pd.Series(
        [20, 22, 24, 20, 22],
        name="Forecast"
    )

    assert forecast.equals(expected)

    # Check output name
    assert forecast.name == "Forecast"
