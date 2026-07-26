import numpy as np
import pandas as pd

#--- Function : stochastic_optimization ---
def stochastic_optimization(scenarios,
                            probabilities,
                            objective_values):
    """
    Evaluate stochastic optimization scenarios using expected value.

    Parameters
    ----------
    scenarios : list
        List of possible scenarios.

    probabilities : list
        Probability associated with each scenario.

    objective_values : array-like
        Objective value obtained for each scenario.

    Returns
    -------
    dict
        Stochastic optimization summary.
    """

    scenarios = np.array(scenarios)

    probabilities = np.array(
        probabilities,
        dtype=float
    )

    objective_values = np.array(
        objective_values,
        dtype=float
    )

    if len(probabilities) != len(objective_values):

        raise ValueError(
            "Probabilities and objective values must have the same length"
        )

    if not np.isclose(
        probabilities.sum(),
        1
    ):

        raise ValueError(
            "Probabilities must sum to 1"
        )

    expected_value = np.sum(
        probabilities *
        objective_values
    )

    variance = np.sum(
        probabilities *
        (
            objective_values
            -
            expected_value
        ) ** 2
    )

    result = {
        "expected_objective_value": expected_value,
        "objective_variance": variance,
        "scenarios": scenarios,
        "probabilities": probabilities,
        "objective_values": objective_values
    }

    print("--- Stochastic Optimization Summary ---")
    print(
        f"Expected objective value : {expected_value:.4f}"
    )
    print(
        f"Objective variance       : {variance:.4f}"
    )
    print("-" * 35)

    return result

#--- Function : robust_optimization ---
def robust_optimization(objective_values,
                        uncertainty_level="worst_case"):
    """
    Evaluate robust optimization performance under uncertainty.

    Parameters
    ----------
    objective_values : array-like
        Objective values obtained under uncertain scenarios.

    uncertainty_level : str, default="worst_case"
        Robustness criterion.

        Options:
        - "worst_case"
        - "best_case"
        - "mean"

    Returns
    -------
    dict
        Robust optimization summary.
    """

    objective_values = np.array(
        objective_values,
        dtype=float
    )

    if uncertainty_level == "worst_case":

        robust_value = objective_values.min()

    elif uncertainty_level == "best_case":

        robust_value = objective_values.max()

    elif uncertainty_level == "mean":

        robust_value = objective_values.mean()

    else:

        raise ValueError(
            "uncertainty_level must be 'worst_case', 'best_case', or 'mean'"
        )

    result = {

        "robust_value": robust_value,

        "uncertainty_level": uncertainty_level,

        "scenario_values": objective_values

    }

    print("--- Robust Optimization Summary ---")
    print(
        f"Criterion : {uncertainty_level}"
    )
    print(
        f"Robust objective value : {robust_value:.4f}"
    )
    print("-" * 35)


    return result
