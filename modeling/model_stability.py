import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

#--- Function : check_clustering_model_stability ---
def check_clustering_model_stability(models, df, predictors):
    """
    Internal Validity Check for Clustering.
    Evaluates one or multiple models using Silhouette, Calinski-Harabasz, and Davies-Bouldin.
    """
    import numpy as np
    import pandas as pd
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

    #English comment: Ensure models is a dictionary for uniform processing
    if not isinstance(models, dict):
        models = {"Model_1": models}

    results = []
    X = df[predictors]

    print(f"{'Model Name':<25} | {'Clusters':<8} | {'Silh.':<7} | {'CH Index':<10} | {'DB Index':<8}")
    print("-" * 75)

    for name, model in models.items():
        #English comment: Extract labels from fitted model or fit if necessary
        if hasattr(model, 'labels_'):
            labels = model.labels_
        else:
            labels = model.fit_predict(X)

        #English comment: Filter noise for density-based models like DBSCAN
        mask = labels != -1
        unique_labels = np.unique(labels[mask])
        n_clusters = len(unique_labels)

        if n_clusters < 2:
            print(f"{name:<25} | Failed: fewer than 2 clusters.")
            continue

        #Computing Core Metrics
        sil = silhouette_score(X[mask], labels[mask])
        ch = calinski_harabasz_score(X[mask], labels[mask])
        db = davies_bouldin_score(X[mask], labels[mask])

        #Status Diagnostic
        status = "EXCELLENT" if sil > 0.50 else "ACCEPTABLE" if sil > 0.25 else "WEAK" if sil > 0 else "INVALID"

        print(f"{name:<25} | {n_clusters:<8} | {sil:<7.4f} | {ch:<10.2f} | {db:<8.4f} ({status})")

        results.append({
            "model_name": name,
            "n_clusters": n_clusters,
            "silhouette": sil,
            "calinski_harabasz": ch,
            "davies_bouldin": db,
            "status": status
        })

    return pd.DataFrame(results).sort_values(by="silhouette", ascending=False)

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
