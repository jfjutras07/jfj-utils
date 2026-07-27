from optimization.uncertainty_optimization import stochastic_optimization

#--- Function : test_stochastic_optimization_basic ---
def test_stochastic_optimization_basic():

    try:

        result = stochastic_optimization(
            scenarios=["Low", "High"],
            probabilities=[0.5, 0.5],
            objective_values=[100, 200]
        )

        assert result["expected_objective_value"] == 150

    except Exception:

        assert False, "Stochastic optimization failed"

#--- Function : test_stochastic_optimization_invalid_probabilities_raises ---
def test_stochastic_optimization_invalid_probabilities_raises():

    try:

        stochastic_optimization(
            scenarios=["Low", "High"],
            probabilities=[0.4, 0.4],
            objective_values=[100, 200]
        )

        assert False, "ValueError should have been raised"

    except ValueError:

        assert True
