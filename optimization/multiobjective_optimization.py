import numpy as np
import pandas as pd

#--- Function : pareto_optimization ---
def pareto_optimization(df,
                        objectives):
    """
    Identify Pareto-optimal solutions from a multi-objective solution set.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset containing candidate solutions and objective values.

    objectives : dict
        Dictionary defining objective directions.
        Example:
        {
            "cost": "min",
            "quality": "max"
        }

    Returns
    -------
    pandas.DataFrame
        Pareto-optimal solutions.
    """

    missing_columns = (
        set(objectives.keys())
        -
        set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing objective columns: {missing_columns}"
        )

    data = df.copy()

    # Convert objectives to maximization form
    values = data[
        list(objectives.keys())
    ].astype(float).copy()

    for objective, direction in objectives.items():

        if direction == "min":

            values[objective] = (
                -values[objective]
            )

        elif direction != "max":

            raise ValueError(
                "Objective direction must be 'min' or 'max'"
            )

    values = values.values

    n_solutions = len(values)

    dominated = np.zeros(
        n_solutions,
        dtype=bool
    )

    # Pareto dominance check
    for i in range(n_solutions):

        for j in range(n_solutions):

            if i == j:
                continue

            if (
                np.all(
                    values[j] >= values[i]
                )
                and
                np.any(
                    values[j] > values[i]
                )
            ):

                dominated[i] = True
                break

    pareto_front = data[
        ~dominated
    ].copy()

    print("--- Pareto Optimization Summary ---")
    print(
        f"Total solutions       : {n_solutions}"
    )
    print(
        f"Pareto optimal points : {len(pareto_front)}"
    )

    print("-" * 35)

    return pareto_front
