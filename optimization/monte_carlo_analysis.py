import numpy as np
import pandas as pd

# --- Function : monte_carlo_simulation ---
def monte_carlo_simulation(model_function,
                           distributions,
                           n_simulations=10000,
                           random_state=42):
    """
    Perform a generic Monte Carlo simulation.

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

    Returns
    -------
    pandas.DataFrame
        Simulated results.
    """

    np.random.seed(random_state)

    results = []

    for _ in range(n_simulations):

        inputs = {
            name: distribution()
            for name, distribution in distributions.items()
        }

        results.append(model_function(**inputs))

    simulation_df = pd.DataFrame({
        "Simulation": range(1, n_simulations + 1),
        "Result": results
    })

    print("--- Monte Carlo Simulation Summary ---")
    print(f"Number of simulations : {n_simulations}")
    print(f"Mean                  : {simulation_df['Result'].mean():.4f}")
    print(f"Std. Dev.             : {simulation_df['Result'].std():.4f}")
    print(f"Minimum               : {simulation_df['Result'].min():.4f}")
    print(f"Maximum               : {simulation_df['Result'].max():.4f}")
    print("-" * 40)

    return simulation_df
