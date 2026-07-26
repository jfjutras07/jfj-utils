import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.seasonal import STL

#--- Function : adf_stationarity_test ---
def adf_stationarity_test(
    df,
    date_col,
    value_col,
    significance_level=0.05,
    regression="c",
    autolag="AIC"
):
    """
    Perform Augmented Dickey-Fuller (ADF) stationarity test
    on a time series.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing the time series.

    date_col : str
        Date column used to order the time series.

    value_col : str
        Numeric variable representing the time series values.

    significance_level : float, default=0.05
        Threshold used for statistical significance.

    regression : str, default="c"
        Regression component used in ADF test.
        Options:
            "c"   : constant
            "ct"  : constant and trend
            "ctt" : constant, trend and quadratic trend
            "n"   : no constant

    autolag : str, default="AIC"
        Method used to select lag length.

    Returns
    -------
    dict
        ADF statistics, critical values,
        and stationarity conclusion.
    """

    # Validate columns
    if date_col not in df.columns:
        raise ValueError(f"'{date_col}' not found in dataframe.")

    if value_col not in df.columns:
        raise ValueError(f"'{value_col}' not found in dataframe.")

    # Prepare time series
    ts = (
        df[[date_col, value_col]]
        .dropna()
        .copy()
    )

    ts[date_col] = pd.to_datetime(
        ts[date_col],
        errors="coerce"
    )

    ts = (
        ts
        .dropna(subset=[date_col])
        .sort_values(date_col)
        .groupby(date_col, as_index=False)
        .agg({value_col: "mean"})
    )

    values = ts[value_col]

    if len(values) < 10:
        raise ValueError(
            "Time series must contain at least 10 observations."
        )

    # ADF test
    result = adfuller(
        values,
        regression=regression,
        autolag=autolag
    )

    adf_statistic = result[0]
    p_value = result[1]
    critical_values = result[4]

    stationary = p_value < significance_level

    return {
        "ADF Statistic": adf_statistic,
        "p-value": p_value,
        "Critical Value 1%": critical_values["1%"],
        "Critical Value 5%": critical_values["5%"],
        "Critical Value 10%": critical_values["10%"],
        "Stationary": stationary,
        "Conclusion": (
            "Series is stationary"
            if stationary
            else "Series is not stationary"
        ),
        "Observations": len(values)
    }

#--- Function : ljung_box_test ---
def ljung_box_test(
    df,
    date_col,
    value_col,
    lags=10,
    significance_level=0.05
):
    """
    Perform Ljung-Box test to evaluate autocorrelation
    in a time series.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing the time series.

    date_col : str
        Date column used to order the time series.

    value_col : str
        Numeric variable representing the time series values.

    lags : int, default=10
        Number of autocorrelation lags tested.

    significance_level : float, default=0.05
        Threshold used for statistical significance.

    Returns
    -------
    dict
        Ljung-Box statistics and autocorrelation conclusion.
    """

    # Validate columns
    if date_col not in df.columns:
        raise ValueError(f"'{date_col}' not found in dataframe.")

    if value_col not in df.columns:
        raise ValueError(f"'{value_col}' not found in dataframe.")

    if not isinstance(lags, int) or lags < 1:
        raise ValueError("'lags' must be a positive integer.")

    # Prepare time series
    ts = (
        df[[date_col, value_col]]
        .dropna()
        .copy()
    )

    ts[date_col] = pd.to_datetime(
        ts[date_col],
        errors="coerce"
    )

    ts = (
        ts
        .dropna(subset=[date_col])
        .sort_values(date_col)
        .groupby(date_col, as_index=False)
        .agg({value_col: "mean"})
    )

    values = ts[value_col]

    if len(values) < lags + 5:
        raise ValueError(
            "Time series length is insufficient for selected lag value."
        )

    # Ljung-Box test
    result = acorr_ljungbox(
        values,
        lags=[lags],
        return_df=True
    )

    statistic = result["lb_stat"].iloc[0]
    p_value = result["lb_pvalue"].iloc[0]

    autocorrelation_present = p_value < significance_level

    return {
        "Lag": lags,
        "Ljung-Box Statistic": statistic,
        "p-value": p_value,
        "Autocorrelation Detected": autocorrelation_present,
        "Conclusion": (
            "Significant temporal dependency detected"
            if autocorrelation_present
            else "No significant autocorrelation detected"
        ),
        "Observations": len(values)
    }

#--- Function : seasonal_strength_test ---
def seasonal_strength_test(
    df,
    date_col,
    value_col,
    period=7
):
    """
    Estimate seasonal strength of a time series
    using STL decomposition.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing the time series.

    date_col : str
        Date column used to order the time series.

    value_col : str
        Numeric variable representing the time series values.

    period : int, default=7
        Seasonal period length.
        Examples:
            7  -> weekly seasonality in daily data
            12 -> yearly seasonality in monthly data
            24 -> daily seasonality in hourly data

    Returns
    -------
    dict
        Seasonal strength indicators and interpretation.
    """

    # Validate columns
    if date_col not in df.columns:
        raise ValueError(f"'{date_col}' not found in dataframe.")

    if value_col not in df.columns:
        raise ValueError(f"'{value_col}' not found in dataframe.")

    if not isinstance(period, int) or period < 2:
        raise ValueError(
            "'period' must be an integer greater than or equal to 2."
        )

    # Prepare time series
    ts = (
        df[[date_col, value_col]]
        .dropna()
        .copy()
    )

    ts[date_col] = pd.to_datetime(
        ts[date_col],
        errors="coerce"
    )

    ts = (
        ts
        .dropna(subset=[date_col])
        .sort_values(date_col)
        .groupby(date_col, as_index=False)
        .agg({value_col: "mean"})
    )

    values = ts[value_col]

    if len(values) < 2 * period:
        raise ValueError(
            "Time series length is insufficient for STL decomposition."
        )

    # STL decomposition
    decomposition = STL(
        values,
        period=period,
        robust=True
    ).fit()

    seasonal = decomposition.seasonal
    residual = decomposition.resid

    denominator = (seasonal + residual).var()

    if denominator == 0:
        seasonal_strength = 0
    else:
        seasonal_strength = max(
            0,
            1 - residual.var() / denominator
        )

    return {
        "Seasonal Period": period,
        "Seasonal Variance": seasonal.var(),
        "Residual Variance": residual.var(),
        "Seasonal Strength": seasonal_strength,
        "Interpretation": (
            "Strong seasonal pattern"
            if seasonal_strength >= 0.6
            else
            "Moderate seasonal pattern"
            if seasonal_strength >= 0.3
            else
            "Weak or no seasonal pattern"
        ),
        "Observations": len(values)
    }
