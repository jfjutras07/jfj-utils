import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds


# --- Function : integer_programming ---
def integer_programming(objective,
                        constraints_matrix,
                        constraints_values,
                        bounds=None,
                        maximize=True):
    """
    Solve an integer programming optimization problem.

    Parameters
    ----------
    objective : list
        Objective function coefficients.

    constraints_matrix : array-like
        Constraint coefficients.

    constraints_values : array-like
        Constraint limits.

    bounds : tuple, optional
        Lower and upper bounds.

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

    if bounds is None:
        bounds = Bounds(
            np.zeros(len(objective)),
            np.full(len(objective), np.inf)
        )

    constraints = LinearConstraint(
        constraints_matrix,
        -np.inf,
        constraints_values
    )

    result = milp(
        c=objective,
        integrality=np.ones(len(objective)),
        bounds=bounds,
        constraints=constraints
    )

    solution = {
        "success": result.success,
        "status": result.message,
        "optimal_values": result.x,
        "objective_value": -result.fun if maximize else result.fun
    }

    print("--- Integer Programming Summary ---")
    print(f"Optimization successful : {result.success}")
    print(f"Objective value         : {solution['objective_value']:.4f}")
    print(f"Optimal variables       : {solution['optimal_values']}")
    print("-" * 35)
