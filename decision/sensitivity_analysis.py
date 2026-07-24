import numpy as np
import pandas as pd

#--- Function : decision_sensitivity_analysis ---
def decision_sensitivity_analysis(df, criteria_weights, benefit_criteria, weight_variations=0.1):
    """
    Analyze decision ranking sensitivity by varying criteria weights.

    Parameters
    ----------
    df : pandas.DataFrame
        Decision matrix.

    criteria_weights : dict
        Original criteria weights.

    benefit_criteria : list
        Criteria where higher values are preferred.

    weight_variations : float
        Percentage variation applied to criteria weights.
    """

    base_weights = pd.Series(criteria_weights, dtype=float)

    results = []

    for criterion in base_weights.index:

        modified_weights = base_weights.copy()

        modified_weights[criterion] = (
            modified_weights[criterion] * (1 + weight_variations)
        )

        modified_weights = modified_weights / modified_weights.sum()

        criteria = list(modified_weights.index)

        matrix = df[criteria].astype(float)

        normalized = matrix / np.sqrt((matrix ** 2).sum())

        weighted_matrix = normalized * modified_weights

        ideal_best = []
        ideal_worst = []

        for col in criteria:

            if col in benefit_criteria:
                ideal_best.append(weighted_matrix[col].max())
                ideal_worst.append(weighted_matrix[col].min())

            else:
                ideal_best.append(weighted_matrix[col].min())
                ideal_worst.append(weighted_matrix[col].max())

        ideal_best = np.array(ideal_best)
        ideal_worst = np.array(ideal_worst)

        distance_best = np.sqrt(
            ((weighted_matrix.values - ideal_best) ** 2).sum(axis=1)
        )

        distance_worst = np.sqrt(
            ((weighted_matrix.values - ideal_worst) ** 2).sum(axis=1)
        )

        scores = distance_worst / (
            distance_best + distance_worst
        )

        ranking = pd.Series(
            scores,
            index=df.index
        ).sort_values(
            ascending=False
        )

        results.append({
            "Modified_Criterion": criterion,
            "Top_Alternative": ranking.index[0],
            "Top_Score": ranking.iloc[0]
        })

    sensitivity_results = pd.DataFrame(results)

    print("--- Decision Sensitivity Analysis Summary ---")
    print(f"Criteria tested : {len(base_weights)}")
    print(f"Weight variation: ±{weight_variations*100:.0f}%")
    print("\nSensitivity Results:")
    print(sensitivity_results.to_string(index=False))
    print("-" * 45)
