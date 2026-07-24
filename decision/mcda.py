import numpy as np
import pandas as pd

#--- Function : ahp_weights ---
def ahp_weights(pairwise_matrix):
    """
    Calculate criteria weights using Analytic Hierarchy Process (AHP).

    Parameters
    ----------
    pairwise_matrix : pandas.DataFrame or array-like
        Pairwise comparison matrix between criteria.

    Returns
    -------
    Series
        Criteria weights.
    """

    matrix = np.array(pairwise_matrix, dtype=float)

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Pairwise comparison matrix must be square")

    eigenvalues, eigenvectors = np.linalg.eig(matrix)

    max_index = np.argmax(eigenvalues.real)

    weights = np.abs(eigenvectors[:, max_index].real)
    weights = weights / weights.sum()

    criteria_weights = pd.Series(
        weights,
        index=getattr(pairwise_matrix, "index", None)
    )

    print("--- AHP Weights Summary ---")
    print("Criteria weights:")
    print(criteria_weights)
    print("-" * 35)

#--- Function : topsis ---
def topsis(df, criteria_weights, benefit_criteria):
    """
    TOPSIS multi-criteria decision analysis.

    Parameters
    ----------
    df : pandas.DataFrame
        Decision matrix where rows are alternatives and columns are criteria.

    criteria_weights : dict or Series
        Criteria weights.

    benefit_criteria : list
        Criteria where higher values are preferred.

    Returns
    -------
    DataFrame
        Alternatives ranked by TOPSIS score.
    """

    criteria = list(criteria_weights.keys())

    missing_columns = set(criteria) - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing criteria columns: {missing_columns}")

    matrix = df[criteria].astype(float)

    # Normalization
    normalized = matrix / np.sqrt((matrix ** 2).sum())

    # Weight application
    weights = pd.Series(criteria_weights)
    weighted_matrix = normalized * weights

    # Ideal solutions
    ideal_best = []
    ideal_worst = []

    for criterion in criteria:

        if criterion in benefit_criteria:
            ideal_best.append(weighted_matrix[criterion].max())
            ideal_worst.append(weighted_matrix[criterion].min())

        else:
            ideal_best.append(weighted_matrix[criterion].min())
            ideal_worst.append(weighted_matrix[criterion].max())

    ideal_best = np.array(ideal_best)
    ideal_worst = np.array(ideal_worst)

    # Distances
    distance_best = np.sqrt(
        ((weighted_matrix.values - ideal_best) ** 2).sum(axis=1)
    )

    distance_worst = np.sqrt(
        ((weighted_matrix.values - ideal_worst) ** 2).sum(axis=1)
    )

    # TOPSIS score
    scores = distance_worst / (
        distance_best + distance_worst
    )

    result = df.copy()

    result["TOPSIS_Score"] = scores

    result = result.sort_values(
        by="TOPSIS_Score",
        ascending=False
    )

    print("--- TOPSIS Summary ---")
    print(f"Number of alternatives : {len(result)}")
    print(f"Criteria used          : {len(criteria)}")
    print("\nRanking:")
    print(result[["TOPSIS_Score"]].to_string())
    print("-" * 35)
