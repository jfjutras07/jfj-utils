import pandas as pd
import numpy as np
from visualization.explore_continuous import plot_scatter_grid

#--- Function : test_plot_scatter_grid_single_no_group ---
def test_plot_scatter_grid_single_no_group():
    df = pd.DataFrame({
        "x1": np.random.rand(30),
        "y1": np.random.rand(30)
    })

    plot_scatter_grid(
        df=df,
        x_cols=["x1"],
        y_cols=["y1"]
    )

    assert True

#--- Function : test_plot_scatter_grid_single_with_group ---
def test_plot_scatter_grid_single_with_group():
    df = pd.DataFrame({
        "x1": np.random.rand(40),
        "y1": np.random.rand(40),
        "Category": np.random.choice(["A", "B"], size=40)
    })

    plot_scatter_grid(
        df=df,
        x_cols=["x1"],
        y_cols=["y1"],
        group_col="Category"
    )

    assert True

#--- Function : test_plot_scatter_grid_single_with_group_labels ---
def test_plot_scatter_grid_single_with_group_labels():
    df = pd.DataFrame({
        "x1": np.random.rand(40),
        "y1": np.random.rand(40),
        "Category": np.random.choice([0, 1], size=40)
    })

    labels = {0: "Group A", 1: "Group B"}

    plot_scatter_grid(
        df=df,
        x_cols=["x1"],
        y_cols=["y1"],
        group_col="Category",
        group_labels=labels
    )

    assert True

#--- Function : test_plot_scatter_grid_multiple_no_group ---
def test_plot_scatter_grid_multiple_no_group():
    df = pd.DataFrame({
        "x1": np.random.rand(50),
        "y1": np.random.rand(50),
        "x2": np.random.rand(50),
        "y2": np.random.rand(50)
    })

    plot_scatter_grid(
        df=df,
        x_cols=["x1", "x2"],
        y_cols=["y1", "y2"],
        n_cols_per_row=2
    )

    assert True

#--- Function : test_plot_scatter_grid_multiple_with_group ---
def test_plot_scatter_grid_multiple_with_group():
    df = pd.DataFrame({
        "x1": np.random.rand(60),
        "y1": np.random.rand(60),
        "x2": np.random.rand(60),
        "y2": np.random.rand(60),
        "Segment": np.random.choice(["Low", "High"], size=60)
    })

    plot_scatter_grid(
        df=df,
        x_cols=["x1", "x2"],
        y_cols=["y1", "y2"],
        group_col="Segment",
        n_cols_per_row=2
    )

    assert True
