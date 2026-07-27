import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

# --- Function : mixed_integer_programming ---
def mixed_integer_programming(objective,
                              constraints_matrix,
                              constraints_values,
                              integrality=None,
                              bounds=None,
                              maximize=True):
    """
    Solve a Mixed Integer Linear Programming (MILP) optimization problem.

    Parameters
    ----------
    objective : list
        Objective function coefficients.

    constraints_matrix : array-like
        Constraint coefficients.

    constraints_values : array-like
        Constraint limits.

    integrality : list, optional
        Variable integrality definition.

        0 : continuous variable
        1 : integer variable
        2 : semi-continuous variable
        3 : semi-integer variable

        If None, all variables are assumed to be integer.

    bounds : scipy.optimize.Bounds, optional
        Lower and upper bounds for variables.

    maximize : bool, default=True
        Defines maximization or minimization problem.

    Returns
    -------
    dict
        Optimization solution.
    """

    objective = np.array(objective, dtype=float)

    if maximize:
        objective = -objective

    if integrality is None:

        integrality = np.ones(
            len(objective),
            dtype=int
        )

    else:

        integrality = np.array(
            integrality,
            dtype=int
        )

    if bounds is None:

        lower_bounds = np.zeros(
            len(objective)
        )

        upper_bounds = np.full(
            len(objective),
            np.inf
        )

        bounds = Bounds(
            lower_bounds,
            upper_bounds
        )

    constraints = LinearConstraint(
        constraints_matrix,
        -np.inf,
        constraints_values
    )

    result = milp(
        c=objective,
        integrality=integrality,
        bounds=bounds,
        constraints=constraints
    )

    solution = {
        "success": result.success,
        "status": result.message,
        "optimal_values": result.x,
        "objective_value": (
            -result.fun if maximize else result.fun
        )
    }

    print("--- Mixed Integer Programming Summary ---")
    print(f"Optimization successful : {result.success}")
    print(f"Objective value         : {solution['objective_value']:.4f}")
    print(f"Optimal variables       : {solution['optimal_values']}")
    print("-" * 45)

    return solution
