import pandas as pd
import matplotlib.pyplot as plt
from visualization.explore_discrete import plot_discrete_distribution

#--- Function : test_plot_discrete_distribution_basic ---
def test_plot_discrete_distribution_basic():
    df = pd.DataFrame({
        "Dept": ["Sales", "Sales", "IT", "IT", "HR", "HR", "IT"]
    })
    plt.ioff()
    plot_discrete_distribution(df, ["Dept"])
    plt.ion()
    assert "Dept" in df.columns

#--- Function : test_plot_discrete_distribution_with_nan ---
def test_plot_discrete_distribution_with_nan():
    df_nan = pd.DataFrame({
        "Dept": ["Sales", None, "IT", "IT", None, "HR", "IT"]
    })
    plt.ioff()
    plot_discrete_distribution(df_nan, ["Dept"])
    plt.ion()
    assert df_nan["Dept"].isna().sum() > 0

#--- Function : test_plot_discrete_distribution_missing_column ---
def test_plot_discrete_distribution_missing_column():
    df = pd.DataFrame({
        "Dept": ["Sales", "Sales", "IT", "IT", "HR", "HR", "IT"]
    })
    plt.ioff()
    plot_discrete_distribution(df, ["Gender"])  # 'Gender' n'existe pas
    plt.ion()
    # On vérifie juste que la fonction n’a pas crashé
    assert "Dept" in df.columns
