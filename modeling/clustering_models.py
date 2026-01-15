import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, Birch
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from IPython.display import display

#---Function:agglomerative_clustering---
def agglomerative_clustering(df, predictors, k=4, optimize=False, k_range=range(2,8), for_compare=False):
    """
    Agglomerative Hierarchical Clustering with Ward linkage.
    Optimizes the number of clusters based on a combined validation metric.
    """
    X = df[predictors]
    best_k = k
    
    if optimize:
        best_score = -np.inf
        for k_test in k_range:
            pipe = Pipeline([('scaler', RobustScaler()), ('model', AgglomerativeClustering(n_clusters=k_test))])
            labels = pipe.fit_predict(X)
            if len(np.unique(labels)) < 2: continue
            
            sil = silhouette_score(X, labels)
            ch = calinski_harabasz_score(X, labels)
            db = davies_bouldin_score(X, labels)
            score = sil + (ch/1000) - db
            
            if score > best_score:
                best_score, best_k = score, k_test
    
    best_pipe = Pipeline([('scaler', RobustScaler()), ('model', AgglomerativeClustering(n_clusters=best_k))])
    best_pipe.fit(X)
    
    if for_compare: return best_pipe

    print(f"--- Agglomerative Clustering Summary ---")
    print(f"Optimal Clusters: {best_k} | Optimized: {optimize}")
    print("-" * 35)
    return best_pipe

#---Function:birch_clustering---
def birch_clustering(df, predictors, k=4, optimize=False, k_range=range(2,8), for_compare=False):
    """
    BIRCH Clustering (Balanced Iterative Reducing and Clustering using Hierarchies).
    Efficient for large datasets by building a Clustering Feature Tree.
    """
    X = df[predictors]
    best_k = k
    
    if optimize:
        best_score = -np.inf
        for k_test in k_range:
            pipe = Pipeline([('scaler', RobustScaler()), ('model', Birch(n_clusters=k_test))])
            labels = pipe.fit_predict(X)
            if len(np.unique(labels)) < 2: continue
            
            sil = silhouette_score(X, labels)
            score = sil
            if score > best_score:
                best_score, best_k = score, k_test
                
    best_pipe = Pipeline([('scaler', RobustScaler()), ('model', Birch(n_clusters=best_k))])
    best_pipe.fit(X)
    
    if for_compare: return best_pipe

    print(f"--- BIRCH Clustering Summary ---")
    print(f"Clusters: {best_k} | Optimized: {optimize}")
    print("-" * 35)
    return best_pipe

#---Function:compare_clustering_models---
def compare_clustering_models(df, predictors, k=4, optimize=False):
    """
    Executes and compares all clustering models, storing labels in the dataframe.
    Sorted by Silhouette score to identify the best partitioning.
    """
    print(f"Starting Clustering Comparison | Predictors: {len(predictors)}")
    print("-" * 45)

    X = df[predictors]
    # Execution in alphabetical order
    models = {
        'Agglomerative': agglomerative_clustering(df, predictors, k=k, optimize=optimize, for_compare=True),
        'BIRCH': birch_clustering(df, predictors, k=k, optimize=optimize, for_compare=True),
        'DBSCAN': dbscan_clustering(df, predictors, optimize=optimize, for_compare=True),
        'GMM': gaussian_mixture_clustering(df, predictors, k=k, optimize=optimize, for_compare=True),
        'KMeans': kmeans_clustering(df, predictors, k=k, optimize=optimize, for_compare=True),
        'KMedoids': kmedoids_clustering(df, predictors, k=k, optimize=optimize, for_compare=True)
    }

    summary_data = []
    for name, pipe in models.items():
        if pipe is None: continue
        labels = pipe.fit_predict(X)
        df[f'Cluster_{name}'] = labels
        n_clusters = len(np.unique(labels[labels != -1]))
        score = silhouette_score(X, labels) if n_clusters > 1 else 0
        
        summary_data.append({
            "Model": name,
            "Clusters": n_clusters,
            "Silhouette": round(score, 4)
        })

    print("\n--- Final Clustering Comparison (Sorted by Silhouette) ---")
    display(pd.DataFrame(summary_data).sort_values(by="Silhouette", ascending=False))
    return models

#---Function:dbscan_clustering---
def dbscan_clustering(df, predictors, eps=0.5, min_samples=5, optimize=False, for_compare=False):
    """
    DBSCAN Clustering. Identifies clusters of any shape and detects noise (-1).
    """
    X = df[predictors]
    best_eps, best_ms = eps, min_samples
    
    if optimize:
        best_score = -np.inf
        for e in [0.3, 0.5, 0.7]:
            for ms in [5, 10]:
                pipe = Pipeline([('scaler', RobustScaler()), ('model', DBSCAN(eps=e, min_samples=ms))])
                labels = pipe.fit_predict(X)
                mask = labels != -1
                if len(np.unique(labels[mask])) < 2: continue
                sil = silhouette_score(X[mask], labels[mask])
                if sil > best_score:
                    best_score, best_eps, best_ms = sil, e, ms
                    
    best_pipe = Pipeline([('scaler', RobustScaler()), ('model', DBSCAN(eps=best_eps, min_samples=best_ms))])
    best_pipe.fit(X)
    
    if for_compare: return best_pipe

    print(f"--- DBSCAN Clustering Summary ---")
    print(f"Eps: {best_eps} | Min Samples: {best_ms} | Optimized: {optimize}")
    print("-" * 35)
    return best_pipe

#---Function:gaussian_mixture_clustering---
def gaussian_mixture_clustering(df, predictors, k=4, optimize=False, k_range=range(2,8), for_compare=False):
    """
    Gaussian Mixture Model (GMM). Uses Expectation-Maximization for probabilistic assignments.
    """
    X = df[predictors]
    best_k = k
    
    if optimize:
        best_score = -np.inf
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        for k_test in k_range:
            gmm = GaussianMixture(n_components=k_test, random_state=42)
            labels = gmm.fit_predict(X_scaled)
            sil = silhouette_score(X_scaled, labels)
            score = -gmm.bic(X_scaled) + (sil * 100) # Balancing BIC and Silhouette
            if score > best_score:
                best_score, best_k = score, k_test
                
    best_pipe = Pipeline([('scaler', RobustScaler()), ('model', GaussianMixture(n_components=best_k, random_state=42))])
    best_pipe.fit(X)
    
    if for_compare: return best_pipe

    print(f"--- GMM Clustering Summary ---")
    print(f"Components: {best_k} | Optimized: {optimize}")
    print("-" * 35)
    return best_pipe

#---Function:kmeans_clustering---
def kmeans_clustering(df, predictors, k=4, optimize=False, k_range=range(2,8), for_compare=False):
    """
    K-Means Clustering. Minimizes within-cluster sum-of-squares (inertia).
    """
    X = df[predictors]
    best_k = k
    
    if optimize:
        best_score = -np.inf
        for k_test in k_range:
            pipe = Pipeline([('scaler', RobustScaler()), ('model', KMeans(n_clusters=k_test, n_init=10, random_state=42))])
            labels = pipe.fit_predict(X)
            sil = silhouette_score(X, labels)
            db = davies_bouldin_score(X, labels)
            score = sil - db
            if score > best_score:
                best_score, best_k = score, k_test
                
    best_pipe = Pipeline([('scaler', RobustScaler()), ('model', KMeans(n_clusters=best_k, n_init=10, random_state=42))])
    best_pipe.fit(X)
    
    if for_compare: return best_pipe

    print(f"--- KMeans Clustering Summary ---")
    print(f"Clusters: {best_k} | Optimized: {optimize}")
    print("-" * 35)
    return best_pipe

#---Function:kmedoids_clustering---
def kmedoids_clustering(df, predictors, k=4, optimize=False, k_range=range(2,8), for_compare=False):
    """
    K-Medoids Clustering. More robust to outliers than K-Means by using actual data points as centers.
    """
    try:
        from sklearn_extra.cluster import KMedoids
    except ImportError:
        return None

    X = df[predictors]
    best_k = k
    
    if optimize:
        best_score = -np.inf
        for k_test in k_range:
            pipe = Pipeline([('scaler', RobustScaler()), ('model', KMedoids(n_clusters=k_test, random_state=42, method='pam'))])
            labels = pipe.fit_predict(X)
            sil = silhouette_score(X, labels)
            if sil > best_score:
                best_score, best_k = sil, k_test
                
    best_pipe = Pipeline([('scaler', RobustScaler()), ('model', KMedoids(n_clusters=best_k, random_state=42, method='pam'))])
    best_pipe.fit(X)
    
    if for_compare: return best_pipe

    print(f"--- K-Medoids Clustering Summary ---")
    print(f"Clusters: {best_k} | Optimized: {optimize}")
    print("-" * 35)
    return best_pipe
