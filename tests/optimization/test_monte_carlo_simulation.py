import numpy as np
import pandas as pd

from optimization.monte_carlo_analysis import monte_carlo_simulation

#--- Function : test_monte_carlo_simulation_basic ---
def test_monte_carlo_simulation_basic():

    def model(cost, benefit):
        return benefit - cost

    distributions = {
        "cost": lambda: np.random.normal(100, 10),
        "benefit": lambda: np.random.normal(150, 15)
    }

    result = monte_carlo_simulation(
        model_function=model,
        distributions=distributions,
        n_simulations=100
    )

    # Validate returned structure
    assert isinstance(result, dict)

    assert "simulation_results" in result
    assert "summary" in result
    assert "sensitivity_analysis" in result

    # Validate simulation output
    simulation_df = result["simulation_results"]

    assert isinstance(
        simulation_df,
        pd.DataFrame
    )

    assert len(simulation_df) == 100

    assert "Result" in simulation_df.columns

    assert "Simulation" in simulation_df.columns

    # Validate summary statistics
    summary = result["summary"]

    assert isinstance(
        summary,
        pd.Series
    )

    expected_metrics = [
        "Mean",
        "Median",
        "Std_Dev",
        "Minimum",
        "Maximum",
        "P5",
        "P25",
        "P75",
        "P95"
    ]

    for metric in expected_metrics:
        assert metric in summary.index

    # Validate sensitivity analysis
    sensitivity = result["sensitivity_analysis"]

    assert isinstance(
        sensitivity,
        pd.Series
    )

    assert "cost" in sensitivity.index
    assert "benefit" in sensitivity.index
