import pandas as pd
import matplotlib.pyplot as plt
from visualization.explore_binary import plot_binary_distribution

#--- Function : test_plot_binary_distribution_basic ---
def test_plot_binary_distribution_basic():
    df = pd.DataFrame({
        "IsManager": [1, 0, 0, 1, 0, 1, 0]
    })
    plt.ioff()
    plot_binary_distribution(df, ["IsManager"])
    plt.ion()
    assert "IsManager" in df.columns

#--- Function : test_plot_binary_distribution_with_nan ---
def test_plot_binary_distribution_with_nan():
    df_nan = pd.DataFrame({
        "IsManager": [1, None, 0, 1, None, 1, 0]
    })
    plt.ioff()
    plot_binary_distribution(df_nan, ["IsManager"])
    plt.ion()
    assert df_nan["IsManager"].isna().sum() > 0

#--- Function : test_plot_binary_distribution_missing_column ---
def test_plot_binary_distribution_missing_column():
    df = pd.DataFrame({
        "IsManager": [1, 0, 0, 1, 0, 1, 0]
    })
    plt.ioff()
    plot_binary_distribution(df, ["IsExecutive"])  # 'IsExecutive' n'existe pas
    plt.ion()
    # On vérifie juste que la fonction n’a pas crashé
    assert "IsManager" in df.columns
