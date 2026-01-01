import pandas as pd
import matplotlib.pyplot as plt
from visualization.explore_continuous import plot_numeric_distribution

#--- Function : test_plot_numeric_distribution_basic ---
def test_plot_numeric_distribution_basic():
    df = pd.DataFrame({
        "age": [25, 30, 22, 40, 35],
        "salary": [50000, 60000, 45000, 80000, 75000]
    })
    plt.ioff()
    plot_numeric_distribution(df, ["age", "salary"])
    plt.ion()
    assert all(col in df.columns for col in ["age", "salary"])

#--- Function : test_plot_numeric_distribution_with_missing_column ---
def test_plot_numeric_distribution_with_missing_column():
    df = pd.DataFrame({"age": [25, 30, 22]})
    plt.ioff()
    plot_numeric_distribution(df, ["age", "height"])  # 'height' n'existe pas
    plt.ion()
    assert "age" in df.columns

#--- Function : test_plot_numeric_distribution_with_nans ---
def test_plot_numeric_distribution_with_nans():
    df = pd.DataFrame({"age": [25, None, 22, None, 35]})
    plt.ioff()
    plot_numeric_distribution(df, ["age"])
    plt.ion()
    assert df["age"].isna().sum() == 2

#--- Function : test_plot_numeric_distribution_empty_list ---
def test_plot_numeric_distribution_empty_list():
    df = pd.DataFrame({"age": [25, 30, 22]})
    plt.ioff()
    plot_numeric_distribution(df, [])  # liste vide
    plt.ion()
    assert df.shape[1] == 1
