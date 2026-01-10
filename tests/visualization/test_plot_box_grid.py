import pandas as pd
import numpy as np
from visualization.explore_continuous import plot_box_grid

#--- Function : test_plot_box_grid_basic ---
def test_plot_box_grid_basic():
    df = pd.DataFrame({
        "Economic_status": np.random.choice(["Low", "Medium", "High"], size=30),
        "income": np.random.rand(30)
    })

    plot_box_grid(
        df=df,
        value_cols=["income"],
        group_col="Economic_status",
        n_rows=1,
        n_cols=1
    )

    assert True

#--- Function : test_plot_box_grid_multiple_groups ---
def test_plot_box_grid_multiple_groups():
    df = pd.DataFrame({
        "Economic_status": np.random.choice(["Low", "Medium", "High"], size=40),
        "Region": np.random.choice(["North", "South"], size=40),
        "income": np.random.rand(40)
    })

    plot_box_grid(
        df=df,
        value_cols=["income"],
        group_col=["Economic_status", "Region"],
        n_rows=1,
        n_cols=2
    )

    assert True

#--- Function : test_plot_box_grid_with_hue ---
def test_plot_box_grid_with_hue():
    df = pd.DataFrame({
        "Economic_status": np.random.choice(["Low", "Medium", "High"], size=50),
        "Gender": np.random.choice(["M", "F"], size=50),
        "income": np.random.rand(50)
    })

    plot_box_grid(
        df=df,
        value_cols=["income"],
        group_col="Economic_status",
        hue_col="Gender",
        n_rows=1,
        n_cols=1
    )

    assert True
