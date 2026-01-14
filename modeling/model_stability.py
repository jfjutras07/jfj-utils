import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

#---Function : check_clustering_model_stability ---
def check_clustering_model_stability(model, df, predictors, seeds=[0, 21, 42, 84], subsample_frac=0.8):
    """
    Algorithmic Stability Check for Clustering.
    Evaluates robustness of cluster assignments under randomness and data perturbation.

    Stability dimensions:
    - Random initialization sensitivity (ARI across seeds)
    - Data perturbation sensitivity (ARI under subsampling)

    Parameters:
    -----------
    model : clustering estimator
        Unfitted clustering model (KMeans, GMM, Agglomerative, etc.).
    df : pd.DataFrame
        Dataset containing clustering features.
    predictors : list
        List of feature names used for clustering.
    seeds : list
        Random seeds used to evaluate initialization stability.
    subsample_frac : float
        Fraction of data used for subsampling stability.
    """

    X = df[predictors].copy()

    # Containers
    labels_by_seed = {}
    ari_seed_scores = []
    ari_subsample_scores = []

    # Random seed stability
    for seed in seeds:
        try:
            model_temp = model.__class__(**model.get_params())
            if hasattr(model_temp, "random_state"):
                model_temp.set_params(random_state=seed)
            labels = model_temp.fit_predict(X)
            labels_by_seed[seed] = labels
        except Exception as e:
            print(f"Seed {seed} failed: {e}")

    base_seed = seeds[0]
    base_labels = labels_by_seed[base_seed]

    for seed in seeds[1:]:
        ari = adjusted_rand_score(base_labels, labels_by_seed[seed])
        ari_seed_scores.append(ari)

    # Subsampling stability
    base_idx = None
    base_sub_labels = None

    for seed in seeds:
        X_sub = X.sample(frac=subsample_frac, random_state=seed)
        model_temp = model.__class__(**model.get_params())
        if hasattr(model_temp, "random_state"):
            model_temp.set_params(random_state=seed)

        labels_sub = model_temp.fit_predict(X_sub)

        if base_idx is None:
            base_idx = X_sub.index
            base_sub_labels = labels_sub
        else:
            common_idx = base_idx.intersection(X_sub.index)
            ari = adjusted_rand_score(
                base_sub_labels[np.isin(base_idx, common_idx)],
                labels_sub[np.isin(X_sub.index, common_idx)]
            )
            ari_subsample_scores.append(ari)

    # Reporting
    print("\n--- Clustering Algorithmic Stability Check ---")
    print(f"Model Class              : {model.__class__.__name__}")
    print(f"Seeds Evaluated           : {seeds}")
    print(f"Subsample Fraction        : {subsample_frac}")
    print("-" * 50)

    print("Random Initialization Stability (ARI):")
    for seed, ari in zip(seeds[1:], ari_seed_scores):
        print(f"  ARI(seed {base_seed} vs {seed}) = {ari:.3f}")
    print(f"  Mean ARI (seeds)        = {np.mean(ari_seed_scores):.3f}")
    print("-" * 50)

    print("Subsampling Stability (ARI):")
    for i, ari in enumerate(ari_subsample_scores):
        print(f"  ARI(subsample run {i+1}) = {ari:.3f}")
    print(f"  Mean ARI (subsampling)  = {np.mean(ari_subsample_scores):.3f}")
    print("-" * 50)

    # Formal stability diagnostic
    mean_ari = np.mean(ari_seed_scores + ari_subsample_scores)

    if mean_ari > 0.85:
        print("Status: HIGHLY STABLE. Cluster structure is robust.")
    elif mean_ari > 0.65:
        print("Status: MODERATELY STABLE. Minor sensitivity detected.")
    elif mean_ari > 0.40:
        print("Status: WEAK STABILITY. Model sensitive to perturbations.")
    else:
        print("Status: UNSTABLE. Cluster assignments are unreliable.")

    print("-" * 50)

    return {
        "ari_seeds_mean": np.mean(ari_seed_scores),
        "ari_subsampling_mean": np.mean(ari_subsample_scores),
        "ari_overall_mean": mean_ari,
        "ari_seeds": ari_seed_scores,
        "ari_subsampling": ari_subsample_scores
    }

#--- Function : check_regression_model_stability ---
def check_regression_model_stability(model, X, y, cv=5, scoring='r2'):
    """
    Algorithmic Stability Check.
    Measures the variance of model performance across multiple folds to 
    validate consistency and detect overfitting.

    Parameters:
    -----------
    model : fitted estimator or Pipeline
        The champion model or pipeline to evaluate.
    X : pd.DataFrame or np.array
        Feature matrix.
    y : pd.Series or np.array
        Target variable.
    cv : int
        Number of folds for cross-validation.
    scoring : str
        Metric to evaluate (e.g., 'r2', 'neg_mean_absolute_error').
    """
    
    #Execution of Cross-Validation
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    
    #Standardizing negative metrics (MAE/RMSE) for display
    if scoring.startswith('neg_'):
        scores = -scores
        metric_label = scoring.replace('neg_', '').upper()
    else:
        metric_label = scoring.upper()

    mean_score = np.mean(scores)
    std_score = np.std(scores)
    
    #Coefficient of Variation (Stability Ratio)
    #A low ratio indicates a robust model that generalizes well.
    stability_ratio = (std_score / mean_score) if mean_score != 0 else 0

    print(f"--- Algorithmic Stability Check ({metric_label}) ---")
    print(f"Number of Folds : {cv}")
    print(f"Scores per Fold : {np.round(scores, 4)}")
    print(f"Mean Score      : {mean_score:.4f}")
    print(f"Std Deviation   : {std_score:.4f}")
    print(f"Stability Ratio : {stability_ratio:.2%}")
    print("-" * 40)

    #Formal Diagnostic
    if stability_ratio > 0.15:
        print("Status: UNSTABLE. High variance detected between folds.")
    elif stability_ratio > 0.05:
        print("Status: CAUTION. Moderate variance observed.")
    else:
        print("Status: STABLE. Model performance is consistent.")
    
    print("-" * 40)
    
    return {
        "mean": mean_score,
        "std": std_score,
        "stability_ratio": stability_ratio
    }
