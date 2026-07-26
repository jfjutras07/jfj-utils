import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import warnings
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import STL
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)
from .style import UNIFORM_BLUE, PALE_PINK

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

#--- Function : plot_acf_pacf ---
def plot_acf_pacf(
    df,
    value_col,
    lags=40,
    method="ywm",
    alpha=0.05,
    figsize=(12, 5),
    title=None
):
    """
    Plot the autocorrelation function (ACF) and partial
    autocorrelation function (PACF) of a time series.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.

    value_col : str
        Numeric variable.

    lags : int, default=40
        Number of lags.

    method : str, default="ywm"
        PACF estimation method.

    alpha : float, default=0.05
        Significance level for confidence intervals.

    figsize : tuple, default=(12, 5)
        Figure size.

    title : str, optional
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
        Generated figure.
    """

    # Validate inputs
    if value_col not in df.columns:
        raise ValueError(f"'{value_col}' not found in dataframe.")

    if not isinstance(lags, int) or lags < 1:
        raise ValueError("'lags' must be a positive integer.")

    # Prepare series
    series = df[value_col].dropna()

    if len(series) < lags + 1:
        raise ValueError(
            "The time series must contain more observations than the number of lags."
        )

    # Plot
    fig, axes = plt.subplots(
        1,
        2,
        figsize=figsize
    )

    plot_acf(
        series,
        lags=lags,
        alpha=alpha,
        ax=axes[0]
    )

    axes[0].set_title("Autocorrelation Function (ACF)")

    plot_pacf(
        series,
        lags=lags,
        alpha=alpha,
        method=method,
        ax=axes[1]
    )

    axes[1].set_title("Partial Autocorrelation Function (PACF)")

    fig.suptitle(
        title or f"ACF and PACF of {value_col}",
        fontsize=14,
        fontweight="bold"
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

#--- Function : plot_distribution_by_period ---
def plot_distribution_by_period(
    df,
    value_cols,
    period_col,
    kind="box",
    facet_col=None,
    figsize=(10, 5),
    title=None
):
    """
    Compare the distribution of one or more numerical variables
    across time periods using boxplots or violin plots.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.

    value_cols : str or list
        Numeric variable(s).

    period_col : str
        Time period grouping variable
        (e.g. Month, Weekday, Quarter, Hour).

    kind : {"box", "violin"}, default="box"
        Type of distribution plot.

    facet_col : str, optional
        Optional grouping variable.

    figsize : tuple, default=(7, 5)
        Figure size.

    title : str, optional
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
        Generated figure.
    """

    # Validate inputs
    if period_col not in df.columns:
        raise ValueError(f"'{period_col}' not found in dataframe.")

    if facet_col is not None and facet_col not in df.columns:
        raise ValueError(f"'{facet_col}' not found in dataframe.")

    if isinstance(value_cols, str):
        value_cols = [value_cols]

    missing_cols = [col for col in value_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Column(s) not found: {missing_cols}")

    if kind not in ["box", "violin"]:
        raise ValueError("'kind' must be either 'box' or 'violin'.")

    # Reshape data
    df_long = df.melt(
        id_vars=[c for c in df.columns if c not in value_cols],
        value_vars=value_cols,
        var_name="Variable",
        value_name="Value"
    )

    facets = (
        [None]
        if facet_col is None
        else sorted(df_long[facet_col].dropna().unique())
    )

    n_facets = len(facets)
    n_rows = math.ceil(n_facets / 2)
    n_cols = min(2, n_facets)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figsize[0] * n_cols, figsize[1] * n_rows),
        squeeze=False
    )

    for idx, facet in enumerate(facets):

        ax = axes[idx // 2, idx % 2]

        plot_data = (
            df_long
            if facet is None
            else df_long[df_long[facet_col] == facet]
        )

        period_series = plot_data[period_col].dropna()

        if pd.api.types.is_categorical_dtype(period_series):
            periods = period_series.cat.categories
        else:
            periods = sorted(period_series.unique())

        values = []
        valid_periods = []

        for p in periods:

            vals = plot_data.loc[
                plot_data[period_col] == p,
                "Value"
            ].dropna()

            if len(vals) > 0:
                values.append(vals)
                valid_periods.append(p)

        if len(values) == 0:
            ax.set_visible(False)
            continue

        if kind == "violin":

            violin = ax.violinplot(
                values,
                showmeans=True,
                showmedians=True
            )

            for body in violin["bodies"]:
                body.set_facecolor(UNIFORM_BLUE)
                body.set_alpha(0.6)

        else:

            bp = ax.boxplot(
                values,
                patch_artist=True
            )

            for box in bp["boxes"]:
                box.set_facecolor(PALE_PINK)

        ax.set_xticks(range(1, len(valid_periods) + 1))
        ax.set_xticklabels(
            valid_periods,
            rotation=45,
            ha="right"
        )

        ax.set_xlabel(period_col)
        ax.set_ylabel("Value")

        ax.set_title(
            str(facet)
            if facet is not None
            else value_cols[0]
        )

    # Remove unused axes
    for i in range(n_facets, n_rows * n_cols):
        fig.delaxes(axes[i // 2, i % 2])

    fig.suptitle(
        title or f"Distribution of {', '.join(value_cols)} by {period_col}",
        fontsize=14,
        fontweight="bold"
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

#--- Function : plot_line_grid_over_time ---
def plot_line_grid_over_time(df, value_cols, time_col='Time', group_col=None, 
                             facet_col=None, agg_func='mean', figsize=(13, 4), 
                             colors=None, title=None):
    """
    Grid of line plots for numeric columns over time. 
    Supports optional grouping and faceting.
    """
    if colors is None:
        colors = [UNIFORM_BLUE, PALE_PINK, 'green', 'orange', 'purple', 'brown']

    if isinstance(value_cols, str):
        value_cols = [value_cols]

    df_long = df.melt(
        id_vars=[c for c in df.columns if c not in value_cols],
        value_vars=value_cols,
        var_name='Variable',
        value_name='Value'
    )

    facets = [None] if facet_col is None else df_long[facet_col].dropna().unique()
    n_facets = len(facets)
    fig, axes = plt.subplots(1, n_facets, figsize=figsize, sharey=True, squeeze=False)
    axes = axes.flatten()

    for i, facet in enumerate(facets):
        ax = axes[i]
        facet_data = df_long if facet is None else df_long[df_long[facet_col] == facet]
        
        group_vars = [time_col]
        if group_col:
            group_vars.append(group_col)
            
        grouped = facet_data.groupby(group_vars)['Value'].agg(agg_func).reset_index()
        
        if group_col:
            for j, level in enumerate(sorted(grouped[group_col].unique())):
                plot_data = grouped[grouped[group_col] == level]
                ax.plot(plot_data[time_col], plot_data['Value'], marker='o', 
                        linewidth=2, label=str(level), color=colors[j % len(colors)])
        else:
            ax.plot(grouped[time_col], grouped['Value'], marker='o', 
                    linewidth=2, color=UNIFORM_BLUE)

        ax.set_title(str(facet) if facet else value_cols[0])
        ax.set_xlabel(time_col)
        if i == 0:
            ax.set_ylabel(agg_func.capitalize())

    if group_col:
        axes[-1].legend(title=group_col, bbox_to_anchor=(1.05, 1), loc='upper left')
        
    main_title = title or f"{', '.join(value_cols)} over {time_col}"
    fig.suptitle(main_title, fontsize=13)
    plt.tight_layout()
    plt.show()

#--- Function : plot_temporal_data ---
def plot_temporal_data(df, value_cols, time_col='Time', group_col=None,
                       facet_col=None, agg_func='mean', rolling_window=None, 
                       show_std=False, title=None, colors=None, figsize=(7, 5)):
    """
    Advanced temporal exploration with rolling averages and standard deviation shading.
    """
    if colors is None:
        colors = [UNIFORM_BLUE, PALE_PINK, 'green', 'orange', 'purple', 'brown']

    if isinstance(value_cols, str):
        value_cols = [value_cols]

    df_long = df.melt(
        id_vars=[c for c in df.columns if c not in value_cols],
        value_vars=value_cols,
        var_name='Variable',
        value_name='Value'
    )

    facets = [None] if facet_col is None else df_long[facet_col].dropna().unique()
    n_facets = len(facets)
    n_rows = math.ceil(n_facets / 2)
    n_cols = min(2, n_facets)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0]*n_cols, figsize[1]*n_rows), squeeze=False)

    for idx, facet in enumerate(facets):
        ax = axes[idx // 2, idx % 2]
        facet_data = df_long if facet is None else df_long[df_long[facet_col] == facet]

        group_vars = [time_col]
        if group_col: group_vars.append(group_col)
        
        stats = facet_data.groupby(group_vars)['Value'].agg([agg_func, 'std']).reset_index()

        if group_col:
            for j, level in enumerate(sorted(stats[group_col].unique())):
                plot_data = stats[stats[group_col] == level].sort_values(time_col)
                y_vals = plot_data[agg_func]
                
                if rolling_window:
                    y_vals = y_vals.rolling(window=rolling_window, center=True).mean()
                
                color = colors[j % len(colors)]
                ax.plot(plot_data[time_col], y_vals, marker='o', linewidth=2, 
                        label=f"{level}", color=color)
                
                if show_std:
                    ax.fill_between(plot_data[time_col], 
                                    plot_data[agg_func] - plot_data['std'],
                                    plot_data[agg_func] + plot_data['std'], 
                                    color=color, alpha=0.15)
        else:
            stats = stats.sort_values(time_col)
            y_vals = stats[agg_func]
            if rolling_window:
                y_vals = y_vals.rolling(window=rolling_window, center=True).mean()
                
            ax.plot(stats[time_col], y_vals, marker='o', color=UNIFORM_BLUE, linewidth=2)
            if show_std:
                ax.fill_between(stats[time_col], 
                                stats[agg_func] - stats['std'],
                                stats[agg_func] + stats['std'], 
                                color=UNIFORM_BLUE, alpha=0.15)

        ax.set_title(str(facet) if facet else 'Signal Analysis')
        ax.set_xlabel(time_col)
        ax.set_ylabel('Value')
        if group_col:
            ax.legend(fontsize=8)

    for i in range(n_facets, n_rows * n_cols):
        fig.delaxes(axes[i // 2, i % 2])

#--- Function : plot_time_diagnostics ---
def plot_time_diagnostics(
    y_true,
    forecasts,
    selection_metric="RMSE",
    title=None,
    figsize=(18, 10)
):
    """
    Display a time series forecasting diagnostic dashboard.

    Includes:
    - Model performance comparison
    - Actual vs best forecast visualization
    - Residual analysis
    - Residual distribution

    Parameters
    ----------
    y_true : pandas.Series
        Actual observed values.

    forecasts : dict
        Dictionary containing model forecasts.

        Example:
        {
            "Naive": forecast_series,
            "ARIMA": forecast_series,
            "Prophet": forecast_series
        }

    selection_metric : {"RMSE", "MAE"}, default="RMSE"
        Metric used to select the best model.

    title : str, optional

    figsize : tuple
        Figure size.

    Returns
    -------
    dict
        Diagnostic summary including metrics and best model.
    """

    if not isinstance(y_true, pd.Series):
        raise TypeError(
            "y_true must be a pandas Series."
        )

    if not isinstance(forecasts, dict):
        raise TypeError(
            "forecasts must be a dictionary."
        )

    if len(forecasts) == 0:
        raise ValueError(
            "forecasts dictionary cannot be empty."
        )

    results = []

    for model_name, forecast in forecasts.items():

        if not isinstance(forecast, pd.Series):
            raise TypeError(
                f"{model_name} forecast must be a pandas Series."
            )

        if len(forecast) != len(y_true):
            raise ValueError(
                f"Length mismatch for model {model_name}."
            )

        rmse = np.sqrt(
            mean_squared_error(
                y_true,
                forecast
            )
        )

        mae = mean_absolute_error(
            y_true,
            forecast
        )

        results.append(
            {
                "Model": model_name,
                "RMSE": rmse,
                "MAE": mae
            }
        )

    metrics_df = (
        pd.DataFrame(results)
        .sort_values(
            by=selection_metric,
            ascending=True
        )
        .reset_index(drop=True)
    )

    best_model = metrics_df.loc[0, "Model"]

    best_forecast = forecasts[best_model]

    residuals = (
        y_true - best_forecast
    )

    # Dashboard
    fig, axes = plt.subplots(
        2,
        2,
        figsize=figsize
    )

    # 1 - Model comparison
    axes[0, 0].bar(
        metrics_df["Model"],
        metrics_df[selection_metric],
        color=PALE_PINK
    )

    axes[0, 0].set_title(
        f"Model Comparison ({selection_metric})"
    )

    axes[0, 0].tick_params(
        axis="x",
        rotation=45
    )


    # 2 - Actual vs best forecast
    axes[0, 1].plot(
        y_true.index,
        y_true,
        label="Actual",
        color=UNIFORM_BLUE,
        linewidth=2
    )

    axes[0, 1].plot(
        best_forecast.index,
        best_forecast,
        label=best_model,
        color=PALE_PINK,
        linewidth=2
    )

    axes[0, 1].set_title(
        f"Best Forecast: {best_model}"
    )

    axes[0, 1].legend()


    # 3 - Residuals over time
    axes[1, 0].plot(
        residuals.index,
        residuals,
        color=UNIFORM_BLUE
    )

    axes[1, 0].axhline(
        0,
        linestyle="--",
        color="black"
    )

    axes[1, 0].set_title(
        "Forecast Residuals Over Time"
    )


    # 4 - Residual distribution
    axes[1, 1].hist(
        residuals,
        bins=30,
        color=PALE_PINK
    )

    axes[1, 1].set_title(
        "Residual Distribution"
    )

    fig.suptitle(
        title or "Time Series Forecasting Diagnostics",
        fontsize=16,
        fontweight="bold"
    )

    plt.tight_layout()
    plt.show()


    return {
        "metrics": metrics_df,
        "best_model": best_model,
        "residuals": residuals
    }

#--- Function : plot_time_decomposition ---
def plot_time_decomposition(
    df,
    date_col,
    value_col,
    period,
    figsize=(12, 9),
    title=None
):
    """
    Perform and visualize STL decomposition of a time series.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.

    date_col : str
        Datetime column.

    value_col : str
        Numeric variable.

    period : int
        Seasonal period.
        Examples:
            7   -> daily data with weekly seasonality
            12  -> monthly data with yearly seasonality
            24  -> hourly data with daily seasonality
            52  -> weekly data with yearly seasonality

    figsize : tuple, default=(12, 9)
        Figure size.

    title : str, optional
        Figure title.

    Returns
    -------
    statsmodels.tsa.seasonal.DecomposeResult
        STL decomposition object containing the observed, trend,
        seasonal, and residual components.
    """

    # Validate inputs
    if date_col not in df.columns:
        raise ValueError(f"'{date_col}' not found in dataframe.")

    if value_col not in df.columns:
        raise ValueError(f"'{value_col}' not found in dataframe.")

    if not isinstance(period, int) or period < 2:
        raise ValueError("'period' must be an integer greater than or equal to 2.")

    # Prepare data
    data = (
        df[[date_col, value_col]]
        .dropna()
        .copy()
    )

    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    data = data.dropna(subset=[date_col])

    data = (
        data
        .groupby(date_col, as_index=False)
        .agg({value_col: "mean"})
        .sort_values(date_col)
        .set_index(date_col)
    )

    if len(data) < 2 * period:
        raise ValueError(
            f"The time series must contain at least {2 * period} observations."
        )

    # STL decomposition
    stl = STL(
        data[value_col],
        period=period,
        robust=True
    )

    result = stl.fit()

    # Plot
    fig, axes = plt.subplots(
        4,
        1,
        figsize=figsize,
        sharex=True
    )

    components = [
        ("Observed", result.observed, UNIFORM_BLUE),
        ("Trend", result.trend, PALE_PINK),
        ("Seasonality", result.seasonal, "green"),
        ("Residuals", result.resid, "gray"),
    ]

    for ax, (label, values, color) in zip(axes, components):
        ax.plot(
            data.index,
            values,
            color=color,
            linewidth=2
        )
        ax.set_title(label)

        if label == "Residuals":
            ax.axhline(
                0,
                color="black",
                linestyle="--",
                linewidth=1
            )

    fig.suptitle(
        title or f"STL Decomposition of {value_col}",
        fontsize=14,
        fontweight="bold"
    )

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

    return result

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
