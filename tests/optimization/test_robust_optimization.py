from optimization.uncertainty_optimization import robust_optimization

#--- Function : test_robust_optimization_basic ---
def test_robust_optimization_basic():

    try:

        result = robust_optimization(
            objective_values=[100, 150, 80],
            uncertainty_level="worst_case"
        )

        assert result["robust_value"] == 80

    except Exception:

        assert False, "Robust optimization failed"

#--- Function : test_robust_optimization_invalid_level_raises ---
def test_robust_optimization_invalid_level_raises():

    try:

        robust_optimization(
            objective_values=[100, 150],
            uncertainty_level="invalid"
        )

        assert False, "ValueError should have been raised"

    except ValueError:

        assert True
