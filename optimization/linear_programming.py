import numpy as np
from scipy.optimize import linprog


# --- Function : linear_programming ---
def linear_programming(objective,
                       constraints_matrix,
                       constraints_values,
                       bounds=None,
                       maximize=True):
    """
    Solve a linear programming optimization problem.

    Parameters
    ----------
    objective : list
        Objective function coefficients.

    constraints_matrix : array-like
        Constraint coefficients.

    constraints_values : array-like
        Constraint limits.

    bounds : list, optional
        Variable bounds.

    maximize : bool, default=True
        Defines maximization or minimization problem.

    Returns
    -------
    dict
        Optimization solution.
    """

    objective = np.array(objective)

    if maximize:
        objective = -objective

    result = linprog(
        c=objective,
        A_ub=constraints_matrix,
        b_ub=constraints_values,
        bounds=bounds,
        method="highs"
    )

    solution = {
        "success": result.success,
        "status": result.message,
        "optimal_values": result.x,
        "objective_value": -result.fun if maximize else result.fun
    }

    print("--- Linear Programming Summary ---")
    print(f"Optimization successful : {result.success}")
    print(f"Objective value         : {solution['objective_value']:.4f}")
    print(f"Optimal variables       : {solution['optimal_values']}")
    print("-" * 35)
