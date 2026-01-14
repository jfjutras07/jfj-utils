import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

#--- Function : check_clustering_model_stability ---
def check_clustering_model_stability(model, df, predictors):
    """
    Internal Validity Check for Clustering.
    Evaluates the cohesion and separation of clusters using mathematical 
    internal metrics.

    Parameters:
    -----------
    model : fitted estimator or Pipeline
        The clustering model (champion) to evaluate.
    df : pd.DataFrame
        DataFrame containing the features.
    predictors : list
        The list of features used during the clustering process.
    """
    X = df[predictors]

    # Extract labels from the model
    # Some models store labels_ after fitting, others require fit_predict
    if hasattr(model, 'labels_'):
        labels = model.labels_
    else:
        labels = model.fit_predict(X)

    # Count actual clusters (excluding noise points -1 if using DBSCAN)
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels[unique_labels != -1])

    if n_clusters < 2:
        print("Performance Check Failed: The model produced fewer than 2 clusters.")
        return None

    # Computing Core Metrics
    sil_score = silhouette_score(X, labels)
    ch_score = calinski_harabasz_score(X, labels)
    db_index = davies_bouldin_score(X, labels)

    print(f"--- Clustering Performance Check ---")
    print(f"Number of Clusters      : {n_clusters}")
    print(f"Silhouette Score (Cohesion) : {sil_score:.4f}  (Goal: -> 1.0)")
    print(f"Calinski-Harabasz Index     : {ch_score:.2f} (Goal: High)")
    print(f"Davies-Bouldin Index        : {db_index:.4f}  (Goal: -> 0.0)")
    print("-" * 45)
    
    # Formal Diagnostic based on Silhouette Score thresholds
    if sil_score > 0.50:
        print("Status: EXCELLENT. Strong and well-separated cluster structure.")
    elif sil_score > 0.25:
        print("Status: ACCEPTABLE. Moderate structure detected; some overlap likely.")
    elif sil_score > 0:
        print("Status: WEAK. Poorly defined clusters; high risk of overlap.")
    else:
        print("Status: INVALID. Model failed to capture a meaningful structure.")
    print("-" * 45)

    return {
        "silhouette": sil_score,
        "calinski_harabasz": ch_score,
        "davies_bouldin": db_index,
        "n_clusters": n_clusters
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
