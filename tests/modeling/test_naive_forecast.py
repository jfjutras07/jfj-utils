import pandas as pd
from modeling.time_series import naive_forecast

#--- Test : naive_forecast ---
def test_naive_forecast():

    train_series = pd.Series(
        [10, 12, 15, 18],
        name="Demand"
    )

    forecast = naive_forecast(
        train_series,
        forecast_horizon=3
    )

    # Check output type
    assert isinstance(
        forecast,
        pd.Series
    )

    # Check forecast length
    assert len(forecast) == 3

    # Check naive logic
    assert all(
        forecast == 18
    )

    # Check output name
    assert forecast.name == "Forecast"
