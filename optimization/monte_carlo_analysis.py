import numpy as np
import pandas as pd


# --- Function : monte_carlo_simulation ---
def monte_carlo_simulation(model_function,
                           distributions,
                           n_simulations=10000,
                           random_state=42,
                           sensitivity_analysis=True):
    """
    Perform a generic Monte Carlo simulation with summary statistics
    and sensitivity analysis.

    Parameters
    ----------
    model_function : callable
        Function returning the simulated outcome.

    distributions : dict
        Dictionary containing input variables and sampling functions.

    n_simulations : int
        Number of Monte Carlo iterations.

    random_state : int
        Random seed for reproducibility.

    sensitivity_analysis : bool
        Calculate input-output correlations.

    Returns
    -------
    dict
        Simulation results, summary statistics and sensitivity analysis.
    """

    np.random.seed(random_state)

    simulation_inputs = []
    results = []

    # Simulation engine
    for _ in range(n_simulations):

        inputs = {
            name: distribution()
            for name, distribution in distributions.items()
        }

        simulation_inputs.append(inputs)

        results.append(
            model_function(**inputs)
        )

    inputs_df = pd.DataFrame(simulation_inputs)

    simulation_df = inputs_df.copy()

    simulation_df["Result"] = results

    simulation_df["Simulation"] = range(
        1,
        n_simulations + 1
    )

    # Summary statistics
    result_series = simulation_df["Result"]

    summary = pd.Series({
        "Mean": result_series.mean(),
        "Median": result_series.median(),
        "Std_Dev": result_series.std(),
        "Minimum": result_series.min(),
        "Maximum": result_series.max(),
        "P5": result_series.quantile(0.05),
        "P25": result_series.quantile(0.25),
        "P75": result_series.quantile(0.75),
        "P95": result_series.quantile(0.95)
    })

    # Sensitivity analysis
    sensitivity = None

    if sensitivity_analysis:

        correlation_matrix = (
            simulation_df.corr()
        )

        sensitivity = (
            correlation_matrix["Result"]
            .drop("Result")
            .sort_values(
                ascending=False
            )
        )

    # Display summary
    print("--- Monte Carlo Simulation Summary ---")
    print(
        f"Number of simulations : {n_simulations}"
    )

    print("\nResult statistics:")
    print(summary)

    if sensitivity_analysis:

        print("\n--- Sensitivity Analysis ---")
        print(sensitivity)

    print("-" * 40)

    return {
        "simulation_results": simulation_df,
        "summary": summary,
        "sensitivity_analysis": sensitivity
    }
