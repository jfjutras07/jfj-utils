import pandas as pd
import matplotlib.pyplot as plt
from visualization.explore_continuous import plot_heatmap_grid

#--- Function : test_plot_heatmap_grid_single_column ---
def test_plot_heatmap_grid_single_column():
    df = pd.DataFrame({
        "Dept": ["Sales", "Sales", "IT", "IT", "HR", "HR"],
        "Salary": [50000, 52000, 60000, 61000, 45000, 46000]
    })
    plt.ioff()
    plot_heatmap_grid(df, value_col="Salary", index_col="Dept")
    plt.ion()
    assert "Dept" in df.columns
    assert "Salary" in df.columns

#--- Function : test_plot_heatmap_grid_two_columns ---
def test_plot_heatmap_grid_two_columns():
    df = pd.DataFrame({
        "Dept": ["Sales", "Sales", "IT", "IT", "HR", "HR"],
        "Gender": ["M", "F", "M", "F", "M", "F"],
        "Salary": [50000, 52000, 60000, 61000, 45000, 46000]
    })
    plt.ioff()
    plot_heatmap_grid(df, value_col="Salary", index_col="Dept", columns_col="Gender")
    plt.ion()
    assert set(["Dept", "Gender", "Salary"]).issubset(df.columns)

#--- Function : test_plot_heatmap_grid_with_nan ---
def test_plot_heatmap_grid_with_nan():
    df_nan = pd.DataFrame({
        "Dept": ["Sales", "Sales", "IT", "IT", "HR", "HR"],
        "Gender": ["M", "F", "M", "F", "M", "F"],
        "Salary": [50000, None, 60000, 61000, None, 46000]
    })
    plt.ioff()
    plot_heatmap_grid(df_nan, value_col="Salary", index_col="Dept", columns_col="Gender")
    plt.ion()
    assert df_nan["Salary"].isna().sum() > 0
