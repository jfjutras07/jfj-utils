import pandas as pd
from modeling.time_series import sarima_forecast

#--- Test : sarima_forecast ---
def test_sarima_forecast():

    train_series = pd.Series(
        [
            20, 22, 21, 25, 28, 30, 29,
            32, 35, 34, 37, 40, 42, 41
        ],
        name="Demand"
    )

    test_series = pd.Series(
        [
            43, 45, 44
        ],
        name="Demand"
    )

    model, forecast, metrics = sarima_forecast(
        train_series,
        test_series,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 7)
    )

    # Check model exists
    assert model is not None

    # Check forecast output
    assert isinstance(
        forecast,
        pd.Series
    )

    assert len(forecast) == len(test_series)

    # Check metrics
    assert isinstance(
        metrics,
        dict
    )

    assert "MAE" in metrics
    assert "RMSE" in metrics
