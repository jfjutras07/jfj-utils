import pandas as pd
from stats.time_series import adf_stationarity_test

#--- Test : adf_stationarity_test ---
def test_adf_stationarity_test():

    df = pd.DataFrame(
        {
            "Date": pd.date_range(
                start="2024-01-01",
                periods=30,
                freq="D"
            ),
            "Demand": [
                20, 21, 19, 22, 18,
                21, 20, 23, 19, 22,
                21, 20, 24, 18, 23,
                22, 19, 21, 20, 24,
                23, 21, 22, 19, 20,
                21, 23, 22, 20, 24
            ]
        }
    )

    result = adf_stationarity_test(
        df,
        date_col="Date",
        value_col="Demand"
    )

    # Check output type
    assert isinstance(
        result,
        pd.Series
    )

    # Check required outputs
    assert "ADF Statistic" in result.index
    assert "p-value" in result.index
    assert "Stationary" in result.index
    assert "Conclusion" in result.index

    # Check logical output type
    assert isinstance(
        result["Stationary"],
        bool
    )
