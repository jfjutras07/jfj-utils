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

#--- Function : fuzzy_ahp_weights ---
def fuzzy_ahp_weights(fuzzy_pairwise_matrix):
    """
    Calculate criteria weights using Fuzzy Analytic Hierarchy Process (Fuzzy AHP).

    Uses triangular fuzzy numbers:
        (l, m, u)

    where:
        l = lower bound
        m = most likely value
        u = upper bound

    Parameters
    ----------
    fuzzy_pairwise_matrix : list or np.array
        Pairwise comparison matrix containing triangular fuzzy numbers.

        Example:
        [
            [(1,1,1), (2,3,4)],
            [(1/4,1/3,1/2), (1,1,1)]
        ]

    Returns
    -------
    pandas.Series
        Defuzzified criteria weights.

    Notes
    -----
    The method:
    1. Calculates fuzzy geometric means.
    2. Normalizes fuzzy weights.
    3. Defuzzifies using centroid method.

    Formula:

    Defuzzification:

        W = (l + m + u) / 3

    """

    matrix = np.array(
        fuzzy_pairwise_matrix,
        dtype=object
    )

    if len(matrix.shape) != 2:
        raise ValueError(
            "Fuzzy pairwise matrix must be two-dimensional."
        )

    n_rows, n_cols = matrix.shape

    if n_rows != n_cols:
        raise ValueError(
            "Fuzzy pairwise matrix must be square."
        )

    fuzzy_weights = []

    # Fuzzy geometric mean
    for i in range(n_rows):

        lower = 1
        middle = 1
        upper = 1

        for j in range(n_cols):

            value = matrix[i][j]

            lower *= value[0]
            middle *= value[1]
            upper *= value[2]

        fuzzy_weights.append(
            (
                lower ** (1/n_cols),
                middle ** (1/n_cols),
                upper ** (1/n_cols)
            )
        )

    # Normalize fuzzy weights

    sum_lower = sum(
        x[0] for x in fuzzy_weights
    )

    sum_middle = sum(
        x[1] for x in fuzzy_weights
    )

    sum_upper = sum(
        x[2] for x in fuzzy_weights
    )

    normalized_weights = []

    for w in fuzzy_weights:

        normalized_weights.append(
            (
                w[0] / sum_upper,
                w[1] / sum_middle,
                w[2] / sum_lower
            )
        )

    # Defuzzification

    crisp_weights = [
        (
            w[0] +
            w[1] +
            w[2]
        ) / 3
        for w in normalized_weights
    ]

    crisp_weights = np.array(
        crisp_weights
    )

    crisp_weights = (
        crisp_weights /
        crisp_weights.sum()
    )

    result = pd.Series(
        crisp_weights,
        name="Fuzzy_AHP_Weight"
    )

    print("--- Fuzzy AHP Weights Summary ---")
    print(result)
    print("-" * 35)

    return result

#--- Function : fuzzy_topsis ---
def fuzzy_topsis(df,
                 criteria_weights,
                 benefit_criteria):
    """
    Fuzzy TOPSIS multi-criteria decision analysis.

    Uses triangular fuzzy numbers:

        (lower, most likely, upper)

    Parameters
    ----------
    df : pandas.DataFrame
        Decision matrix where each value is a triangular fuzzy number.

        Example:

        Partner_A:
            capacity = (7,8,9)

    criteria_weights : dict or Series
        Criteria weights.

    benefit_criteria : list
        Criteria where higher values are preferred.

    Returns
    -------
    pandas.DataFrame
        Alternatives ranked by fuzzy TOPSIS score.

    Method
    ------
    1. Normalize fuzzy decision matrix.
    2. Apply criteria weights.
    3. Determine fuzzy ideal best/worst solutions.
    4. Calculate fuzzy distances.
    5. Compute closeness coefficient.

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


    matrix = df[criteria]

    alternatives = df.index


    # Convert fuzzy numbers to arrays

    fuzzy_matrix = np.array(
        [
            [
                matrix.iloc[i][j]
                for j in range(len(criteria))
            ]
            for i in range(len(matrix))
        ],
        dtype=object
    )


    # Normalize fuzzy matrix

    normalized = np.empty_like(
        fuzzy_matrix
    )

    for j, criterion in enumerate(criteria):

        column = [
            fuzzy_matrix[i][j]
            for i in range(len(matrix))
        ]

        if criterion in benefit_criteria:

            max_value = max(
                x[2]
                for x in column
            )

            for i in range(len(matrix)):

                x = fuzzy_matrix[i][j]

                normalized[i][j] = (
                    x[0] / max_value,
                    x[1] / max_value,
                    x[2] / max_value
                )

        else:

            min_value = min(
                x[0]
                for x in column
            )

            for i in range(len(matrix)):

                x = fuzzy_matrix[i][j]

                normalized[i][j] = (
                    min_value / x[2],
                    min_value / x[1],
                    min_value / x[0]
                )

    # Weighted normalized matrix

    weighted = np.empty_like(
        normalized
    )

    for i in range(len(matrix)):

        for j, criterion in enumerate(criteria):

            weight = criteria_weights[criterion]

            x = normalized[i][j]

            weighted[i][j] = (
                x[0] * weight,
                x[1] * weight,
                x[2] * weight
            )

    # Ideal solutions

    ideal_best = []
    ideal_worst = []

    for j, criterion in enumerate(criteria):

        values = [
            weighted[i][j]
            for i in range(len(matrix))
        ]

        if criterion in benefit_criteria:

            ideal_best.append(
                max(
                    values,
                    key=lambda x: x[1]
                )
            )

            ideal_worst.append(
                min(
                    values,
                    key=lambda x: x[1]
                )
            )

        else:

            ideal_best.append(
                min(
                    values,
                    key=lambda x: x[1]
                )
            )

            ideal_worst.append(
                max(
                    values,
                    key=lambda x: x[1]
                )
            )

    # Vertex distance

    def fuzzy_distance(a, b):

        return np.sqrt(
            (
                (a[0]-b[0])**2 +
                (a[1]-b[1])**2 +
                (a[2]-b[2])**2
            ) / 3
        )

    scores = []

    for i in range(len(matrix)):

        distance_best = 0
        distance_worst = 0

        for j in range(len(criteria)):

            distance_best += fuzzy_distance(
                weighted[i][j],
                ideal_best[j]
            )

            distance_worst += fuzzy_distance(
                weighted[i][j],
                ideal_worst[j]
            )

        score = (
            distance_worst /
            (
                distance_best +
                distance_worst
            )
        )

        scores.append(score)

    result = df.copy()

    result["Fuzzy_TOPSIS_Score"] = scores

    result = result.sort_values(
        by="Fuzzy_TOPSIS_Score",
        ascending=False
    )

    print("--- Fuzzy TOPSIS Summary ---")
    print(
        f"Number of alternatives : {len(result)}"
    )

    print(
        "\nRanking:"
    )

    print(
        result[
            ["Fuzzy_TOPSIS_Score"]
        ].to_string()
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

    # Determine ideal best and worst values
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

                distance = (
                    best[criterion]
                    -
                    row[criterion]
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
