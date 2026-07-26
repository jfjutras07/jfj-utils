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
    pandas.Series
        Criteria weights.
    """

    matrix = np.array(pairwise_matrix, dtype=float)

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            "Pairwise comparison matrix must be square"
        )

    eigenvalues, eigenvectors = np.linalg.eig(matrix)

    max_index = np.argmax(
        eigenvalues.real
    )

    weights = np.abs(
        eigenvectors[:, max_index].real
    )

    weights = weights / weights.sum()

    criteria_weights = pd.Series(
        weights,
        index=getattr(pairwise_matrix, "index", None)
    )

    print("--- AHP Weights Summary ---")
    print("Criteria weights:")
    print(criteria_weights)
    print("-" * 35)

    return criteria_weights

#--- Function : electre_i ---
def electre_i(df,
              criteria_weights,
              benefit_criteria,
              concordance_threshold=0.7):
    """
    ELECTRE I multi-criteria decision analysis.

    Parameters
    ----------
    df : pandas.DataFrame
        Decision matrix where rows are alternatives and columns are criteria.

    criteria_weights : dict or Series
        Criteria weights.

    benefit_criteria : list
        Criteria where higher values are preferred.

    concordance_threshold : float, default=0.7
        Minimum concordance index required for outranking.

    Returns
    -------
    pandas.DataFrame
        ELECTRE outranking matrix.
    """

    criteria = list(
        criteria_weights.keys()
    )

    missing_columns = (
        set(criteria)
        -
        set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing criteria columns: {missing_columns}"
        )

    matrix = df[criteria].astype(float)

    weights = pd.Series(
        criteria_weights
    )

    n_alternatives = len(matrix)

    outranking_matrix = np.zeros(
        (
            n_alternatives,
            n_alternatives
        )
    )

    for i in range(n_alternatives):

        for j in range(n_alternatives):

            if i == j:
                continue

            concordance = 0

            for criterion in criteria:

                if criterion in benefit_criteria:

                    if (
                        matrix.iloc[i][criterion]
                        >=
                        matrix.iloc[j][criterion]
                    ):
                        concordance += weights[criterion]

                else:

                    if (
                        matrix.iloc[i][criterion]
                        <=
                        matrix.iloc[j][criterion]
                    ):
                        concordance += weights[criterion]

            if concordance >= concordance_threshold:

                outranking_matrix[i, j] = 1

    result = pd.DataFrame(
        outranking_matrix,
        index=df.index,
        columns=df.index
    )

    print("--- ELECTRE I Summary ---")
    print(
        f"Number of alternatives : {n_alternatives}"
    )
    print(
        f"Concordance threshold : {concordance_threshold}"
    )

    print("-" * 35)

    return result

#--- Function : promethee_ii ---
def promethee_ii(df,
                 criteria_weights,
                 benefit_criteria):
    """
    PROMETHEE II multi-criteria decision analysis.

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
    pandas.DataFrame
        Alternatives ranked by PROMETHEE net flow.
    """

    criteria = list(
        criteria_weights.keys()
    )

    missing_columns = (
        set(criteria)
        -
        set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing criteria columns: {missing_columns}"
        )

    matrix = df[criteria].astype(float)

    weights = pd.Series(
        criteria_weights
    )

    n_alternatives = len(matrix)

    preference_matrix = np.zeros(
        (
            n_alternatives,
            n_alternatives
        )
    )

    for i in range(n_alternatives):

        for j in range(n_alternatives):

            if i == j:
                continue

            preference = 0

            for criterion in criteria:

                difference = (
                    matrix.iloc[i][criterion]
                    -
                    matrix.iloc[j][criterion]
                )

                if criterion not in benefit_criteria:
                    difference = -difference

                if difference > 0:
                    preference += weights[criterion]

            preference_matrix[i, j] = preference

    positive_flow = (
        preference_matrix.sum(axis=1)
        /
        (n_alternatives - 1)
    )

    negative_flow = (
        preference_matrix.sum(axis=0)
        /
        (n_alternatives - 1)
    )

    net_flow = (
        positive_flow
        -
        negative_flow
    )

    result = df.copy()

    result["PROMETHEE_Net_Flow"] = net_flow

    result = result.sort_values(
        by="PROMETHEE_Net_Flow",
        ascending=False
    )

    print("--- PROMETHEE II Summary ---")
    print(
        f"Number of alternatives : {n_alternatives}"
    )
    print(
        f"Criteria used          : {len(criteria)}"
    )
    print("\nRanking:")

    print(
        result[
            ["PROMETHEE_Net_Flow"]
        ].to_string()
    )

    print("-" * 35)

    return result

#--- Function : topsis ---
def topsis(df,
           criteria_weights,
           benefit_criteria):
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
    pandas.DataFrame
        Alternatives ranked by TOPSIS score.
    """

    criteria = list(
        criteria_weights.keys()
    )

    missing_columns = (
        set(criteria)
        -
        set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing criteria columns: {missing_columns}"
        )

    matrix = df[criteria].astype(float)

    # Vector normalization
    normalized = (
        matrix /
        np.sqrt(
            (matrix ** 2).sum()
        )
    )

    weights = pd.Series(
        criteria_weights
    )

    weighted_matrix = (
        normalized *
        weights
    )

    ideal_best = []
    ideal_worst = []

    for criterion in criteria:

        if criterion in benefit_criteria:

            ideal_best.append(
                weighted_matrix[criterion].max()
            )

            ideal_worst.append(
                weighted_matrix[criterion].min()
            )

        else:

            ideal_best.append(
                weighted_matrix[criterion].min()
            )

            ideal_worst.append(
                weighted_matrix[criterion].max()
            )

    ideal_best = np.array(
        ideal_best
    )

    ideal_worst = np.array(
        ideal_worst
    )

    distance_best = np.sqrt(
        (
            (
                weighted_matrix.values
                -
                ideal_best
            ) ** 2
        ).sum(axis=1)
    )

    distance_worst = np.sqrt(
        (
            (
                weighted_matrix.values
                -
                ideal_worst
            ) ** 2
        ).sum(axis=1)
    )

    scores = (
        distance_worst /
        (
            distance_best
            +
            distance_worst
        )
    )

    result = df.copy()

    result["TOPSIS_Score"] = scores

    result = result.sort_values(
        by="TOPSIS_Score",
        ascending=False
    )

    print("--- TOPSIS Summary ---")
    print(
        f"Number of alternatives : {len(result)}"
    )
    print(
        f"Criteria used          : {len(criteria)}"
    )
    print("\nRanking:")
    print(
        result[
            ["TOPSIS_Score"]
        ].to_string()
    )
    print("-" * 35)

    return result

#--- Function : vikor ---
def vikor(df,
          criteria_weights,
          benefit_criteria,
          v=0.5):
    """
    VIKOR multi-criteria decision analysis.

    Parameters
    ----------
    df : pandas.DataFrame
        Decision matrix where rows are alternatives and columns are criteria.

    criteria_weights : dict or Series
        Criteria weights.

    benefit_criteria : list
        Criteria where higher values are preferred.

    v : float, default=0.5
        Weight of the majority utility strategy.
        v = 0.5 represents a balanced compromise.

    Returns
    -------
    pandas.DataFrame
        Alternatives ranked by VIKOR compromise score.
    """

    criteria = list(
        criteria_weights.keys()
    )

    missing_columns = (
        set(criteria)
        -
        set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing criteria columns: {missing_columns}"
        )

    if not 0 <= v <= 1:
        raise ValueError(
            "v must be between 0 and 1"
        )

    matrix = df[criteria].astype(float)

    weights = pd.Series(
        criteria_weights
    )

    # Determine best and worst values
    best = {}
    worst = {}

    for criterion in criteria:

        if criterion in benefit_criteria:

            best[criterion] = matrix[criterion].max()
            worst[criterion] = matrix[criterion].min()

        else:

            best[criterion] = matrix[criterion].min()
            worst[criterion] = matrix[criterion].max()

    S = []
    R = []

    # Utility and regret calculations
    for index, row in matrix.iterrows():

        utility = 0
        regret = 0

        criterion_distances = []

        for criterion in criteria:

            denominator = (
                best[criterion]
                -
                worst[criterion]
            )

            if denominator == 0:
                distance = 0

            else:

                if criterion in benefit_criteria:

                    distance = (
                        best[criterion]
                        -
                        row[criterion]
                    ) / denominator

                else:

                    distance = (
                        row[criterion]
                        -
                        best[criterion]
                    ) / denominator

            weighted_distance = (
                weights[criterion]
                *
                distance
            )

            utility += weighted_distance

            criterion_distances.append(
                weighted_distance
            )

        regret = max(
            criterion_distances
        )

        S.append(
            utility
        )

        R.append(
            regret
        )

    S = np.array(S)
    R = np.array(R)

    # Normalize S and R
    S_best = S.min()
    S_worst = S.max()

    R_best = R.min()
    R_worst = R.max()

    Q = []

    for i in range(len(matrix)):

        S_term = (
            (S[i] - S_best)
            /
            (S_worst - S_best)
            if S_worst != S_best
            else 0
        )

        R_term = (
            (R[i] - R_best)
            /
            (R_worst - R_best)
            if R_worst != R_best
            else 0
        )

        Q.append(
            v * S_term
            +
            (1 - v) * R_term
        )

    result = df.copy()

    result["VIKOR_S"] = S
    result["VIKOR_R"] = R
    result["VIKOR_Q"] = Q

    result = result.sort_values(
        by="VIKOR_Q",
        ascending=True
    )

    print("--- VIKOR Summary ---")
    print(
        f"Number of alternatives : {len(result)}"
    )
    print(
        f"Criteria used          : {len(criteria)}"
    )
    print(
        f"Compromise parameter v : {v}"
    )

    print("\nRanking:")

    print(
        result[
            ["VIKOR_Q"]
        ].to_string()
    )

    print("-" * 35)

    return result
