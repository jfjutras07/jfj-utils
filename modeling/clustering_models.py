import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, Birch
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import silhouette_score

#---Function:agglomerative_clustering---
def agglomerative_clustering(df, predictors, k=4, optimize=False, k_range=range(2, 8), for_compare=False):
    """
    Agglomerative Hierarchical clustering.
    """
    X = df[predictors]

    if optimize:
        best_score = -1
        best_pipe = None

        for k_test in k_range:
            pipe = Pipeline([
                ('scaler', RobustScaler()),
                ('model', AgglomerativeClustering(n_clusters=k_test, linkage='ward'))
            ])
            labels = pipe.fit_predict(X)
            if len(np.unique(labels)) > 1:
                score = silhouette_score(X, labels)
                if score > best_score:
                    best_score = score
                    best_pipe = pipe
                    k = k_test
        base_pipe = best_pipe
    else:
        base_pipe = Pipeline([
            ('scaler', RobustScaler()),
            ('model', AgglomerativeClustering(n_clusters=k))
        ])
        base_pipe.fit(X)

    if for_compare:
        return base_pipe

    print(f"--- Agglomerative Execution ---")
    print(f"Clusters: {k} | Optimized: {optimize}")
    print("-" * 35)

    return base_pipe

#---Function:birch_clustering---
def birch_clustering(df, predictors, k=4, optimize=False, k_range=range(2, 8), for_compare=False):
    """
    BIRCH clustering - Balanced Iterative Reducing and Clustering using Hierarchies.
    """
    X = df[predictors]

    if optimize:
        best_score = -1
        best_pipe = None

        for k_test in k_range:
            pipe = Pipeline([
                ('scaler', RobustScaler()),
                ('model', Birch(n_clusters=k_test))
            ])
            labels = pipe.fit_predict(X)
            if len(np.unique(labels)) > 1:
                score = silhouette_score(X, labels)
                if score > best_score:
                    best_score = score
                    best_pipe = pipe
                    k = k_test
        base_pipe = best_pipe
    else:
        base_pipe = Pipeline([
            ('scaler', RobustScaler()),
            ('model', Birch(n_clusters=k))
        ])
        base_pipe.fit(X)

    if for_compare:
        return base_pipe

    print(f"--- BIRCH Execution ---")
    print(f"Clusters: {k} | Optimized: {optimize}")
    print("-" * 35)

    return base_pipe

#---Function:compare_clustering_models---
def compare_clustering_models(df, predictors, k=4, optimize=False):
    """
    Executes and compares all clustering models.
    """
    print("Starting Clustering Models Comparison...")
    print(f"Predictors: {len(predictors)} | Initial K: {k} | Optimization: {optimize}")
    print("-" * 45)

    X = df[predictors]
    results = {}

    results['Agglomerative'] = agglomerative_clustering(
        df, predictors, k=k, optimize=optimize
    )

    results['BIRCH'] = birch_clustering(
        df, predictors, k=k, optimize=optimize
    )

    results['DBSCAN'] = dbscan_clustering(
        df, predictors, optimize=optimize
    )

    results['GMM'] = gaussian_mixture_clustering(
        df, predictors, k=k, optimize=optimize
    )

    results['KMeans'] = kmeans_clustering(
        df, predictors, k=k, optimize=optimize
    )

    km_model = kmedoids_clustering(
        df, predictors, k=k, optimize=optimize
    )
    if km_model is not None:
        results['KMedoids'] = km_model

    print("\n--- Summary of Clustering Assignments ---")
    summary_data = []

    for name, model in results.items():
        labels = model.fit_predict(X)
        n_clusters = len(np.unique(labels[labels != -1]))
        df[f'Cluster_{name}'] = labels

        summary_data.append({
            "Model": name,
            "Clusters_Detected": n_clusters
        })

    print(pd.DataFrame(summary_data).to_string(index=False))
    return results

#---Function:dbscan_clustering---
def dbscan_clustering(df, predictors, eps=0.5, min_samples=5, optimize=False,
                      eps_grid=[0.3, 0.5, 0.7], min_samples_grid=[5, 10], for_compare=False):
    """
    DBSCAN clustering - Density-based spatial clustering.
    """
    X = df[predictors]
    best_score = -1
    best_pipe = None

    if optimize:
        for eps in eps_grid:
            for ms in min_samples_grid:
                pipe = Pipeline([
                    ('scaler', RobustScaler()),
                    ('model', DBSCAN(eps=eps, min_samples=ms))
                ])
                labels = pipe.fit_predict(X)
                mask = labels != -1
                if len(np.unique(labels[mask])) > 1:
                    score = silhouette_score(X[mask], labels[mask])
                    if score > best_score:
                        best_score = score
                        best_pipe = pipe
                        eps, min_samples = eps, ms
        base_pipe = best_pipe
    else:
        base_pipe = Pipeline([
            ('scaler', RobustScaler()),
            ('model', DBSCAN(eps=eps, min_samples=min_samples))
        ])
        base_pipe.fit(X)

    if for_compare:
        return base_pipe

    print(f"--- DBSCAN Execution ---")
    print(f"Eps: {eps} | Min Samples: {min_samples} | Optimized: {optimize}")
    print("-" * 35)

    return base_pipe

#---Function:gaussian_mixture_clustering---
def gaussian_mixture_clustering(df, predictors, k=4, optimize=False, k_range=range(2, 8), for_compare=False):
    """
    Gaussian Mixture Models (GMM) - Expectation-Maximization clustering.
    """
    X = df[predictors]
    best_bic = np.inf
    best_pipe = None

    if optimize:
        for k_test in k_range:
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X)
            gmm = GaussianMixture(n_components=k_test, covariance_type='full', random_state=42)
            gmm.fit(X_scaled)
            bic = gmm.bic(X_scaled)
            if bic < best_bic:
                best_bic = bic
                best_pipe = Pipeline([
                    ('scaler', scaler),
                    ('model', gmm)
                ])
                k = k_test
        base_pipe = best_pipe
    else:
        base_pipe = Pipeline([
            ('scaler', RobustScaler()),
            ('model', GaussianMixture(n_components=k, random_state=42))
        ])
        base_pipe.fit(X)

    if for_compare:
        return base_pipe

    print(f"--- GMM Execution ---")
    print(f"Components: {k} | Optimized: {optimize}")
    print("-" * 35)

    return base_pipe

#---Function:kmeans_clustering---
def kmeans_clustering(df, predictors, k=4, optimize=False, k_range=range(2, 8), for_compare=False):
    """
    K-Means clustering algorithm.
    """
    X = df[predictors]
    best_score = -1
    best_pipe = None

    if optimize:
        for k_test in k_range:
            pipe = Pipeline([
                ('scaler', RobustScaler()),
                ('model', KMeans(n_clusters=k_test, random_state=42, n_init=20))
            ])
            labels = pipe.fit_predict(X)
            if len(np.unique(labels)) > 1:
                score = silhouette_score(X, labels)
                if score > best_score:
                    best_score = score
                    best_pipe = pipe
                    k = k_test
        base_pipe = best_pipe
    else:
        base_pipe = Pipeline([
            ('scaler', RobustScaler()),
            ('model', KMeans(n_clusters=k, random_state=42, n_init=10))
        ])
        base_pipe.fit(X)

    if for_compare:
        return base_pipe

    print(f"--- KMeans Execution ---")
    print(f"Clusters: {k} | Optimized: {optimize}")
    print("-" * 35)

    return base_pipe

#---Function:kmedoids_clustering---
def kmedoids_clustering(df, predictors, k=4, optimize=False, k_range=range(2, 8), for_compare=False):
    """
    K-Medoids (PAM) clustering.
    """
    try:
        from sklearn_extra.cluster import KMedoids
    except (ImportError, Exception):
        return None

    X = df[predictors]
    best_score = -1
    best_pipe = None

    if optimize:
        for k_test in k_range:
            pipe = Pipeline([
                ('scaler', RobustScaler()),
                ('model', KMedoids(n_clusters=k_test, random_state=42, method='pam'))
            ])
            labels = pipe.fit_predict(X)
            if len(np.unique(labels)) > 1:
                score = silhouette_score(X, labels)
                if score > best_score:
                    best_score = score
                    best_pipe = pipe
                    k = k_test
        base_pipe = best_pipe
    else:
        base_pipe = Pipeline([
            ('scaler', RobustScaler()),
            ('model', KMedoids(n_clusters=k, random_state=42, method='pam'))
        ])
        base_pipe.fit(X)

    if for_compare:
        return base_pipe

    print(f"--- K-Medoids Execution ---")
    print(f"Clusters: {k} | Optimized: {optimize}")
    print("-" * 35)

    return base_pipe
