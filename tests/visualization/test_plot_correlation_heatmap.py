import pandas as pd
import matplotlib.pyplot as plt
from visualization.explore_continuous import plot_correlation_heatmap

#--- Function : test_plot_correlation_heatmap_basic ---
def test_plot_correlation_heatmap_basic():
    df = pd.DataFrame({
        "age": [25, 30, 22, 40, 35, 28],
        "salary": [50000, 60000, 45000, 80000, 75000, 52000],
        "bonus": [5000, 7000, 3000, 10000, 8000, 6000]
    })
    plt.ioff()
    plot_correlation_heatmap(df)
    plt.ion()
    assert set(["age", "salary", "bonus"]).issubset(df.columns)

#--- Function : test_plot_correlation_heatmap_with_nan ---
def test_plot_correlation_heatmap_with_nan():
    df_nan = pd.DataFrame({
        "age": [25, None, 22, None, 35, 28],
        "salary": [50000, 60000, None, 80000, 75000, 52000],
        "bonus": [5000, 7000, 3000, None, 8000, 6000]
    })
    plt.ioff()
    plot_correlation_heatmap(df_nan)
    plt.ion()
    assert df_nan.isna().sum().sum() > 0

#--- Function : test_plot_correlation_heatmap_specific_columns ---
def test_plot_correlation_heatmap_specific_columns():
    df = pd.DataFrame({
        "age": [25, 30, 22, 40, 35, 28],
        "salary": [50000, 60000, 45000, 80000, 75000, 52000],
        "bonus": [5000, 7000, 3000, 10000, 8000, 6000]
    })
    plt.ioff()
    plot_correlation_heatmap(df, numeric_cols=["age", "salary"])
    plt.ion()
    assert set(["age", "salary"]).issubset(df.columns)
