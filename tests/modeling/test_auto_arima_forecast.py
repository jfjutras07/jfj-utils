import pandas as pd
from modeling.time_models import auto_arima_forecast

#--- Test : auto_arima_forecast ---
def test_auto_arima_forecast():

    train_series = pd.Series(
        [
            20, 22, 21, 25, 28, 30, 29,
            32, 35, 34, 37, 40, 42, 41,
            43, 45, 44, 47, 50, 52, 51
        ],
        name="Demand"
    )

    test_series = pd.Series(
        [53, 55, 54],
        name="Demand"
    )

    result = auto_arima_forecast(
        train_series,
        test_series,
        seasonal=True,
        m=7
    )

    # Output structure validation
    assert isinstance(result, dict)

    assert "model" in result
    assert "forecast" in result
    assert "metrics" in result
    assert "parameters" in result

    # Forecast validation
    assert isinstance(
        result["forecast"],
        pd.Series
    )

    assert len(result["forecast"]) == len(test_series)

    # Metrics validation
    assert "MAE" in result["metrics"]
    assert "RMSE" in result["metrics"]

    # Parameters validation
    assert "order" in result["parameters"]

    print("✅ auto_arima_forecast test passed")
