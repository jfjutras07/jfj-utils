import numpy as np

from optimization.monte_carlo_analysis import monte_carlo_simulation

#--- Function : test_monte_carlo_simulation_basic ---
def test_monte_carlo_simulation_basic():

    def model(cost, benefit):
        return benefit - cost

    distributions = {
        "cost": lambda: np.random.normal(100, 10),
        "benefit": lambda: np.random.normal(150, 15)
    }

    try:
        monte_carlo_simulation(
            model_function=model,
            distributions=distributions,
            n_simulations=100
        )

        assert True

    except Exception:
        assert False, "Monte Carlo simulation failed"
