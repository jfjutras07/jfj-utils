import pandas as pd
from visualization.time_series import plot_time_diagnostics

#--- Test : plot_time_diagnostics ---
def test_plot_time_diagnostics():

    y_true = pd.Series(
        [20, 22, 21, 25, 28],
        name="Demand"
    )

    forecasts = {
        "Naive": pd.Series(
            [19, 21, 22, 24, 27],
            name="Forecast"
        ),
        "SARIMA": pd.Series(
            [20, 21, 22, 26, 28],
            name="Forecast"
        )
    }

    result = plot_time_diagnostics(
        y_true,
        forecasts
    )

    # Check output type
    assert isinstance(
        result,
        dict
    )

    # Check expected outputs
    assert "metrics" in result
    assert "best_model" in result
    assert "residuals" in result

    # Check best model output
    assert result["best_model"] in forecasts

    # Check residual output
    assert isinstance(
        result["residuals"],
        pd.Series
    )
