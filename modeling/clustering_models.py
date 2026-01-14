import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, Birch
from sklearn.mixture import GaussianMixture
from sklearn_extra.cluster import KMedoids
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

#---Function:agglomerative_clustering---
def agglomerative_clustering(df, predictors, k=4, for_compare=False):
    """
    Agglomerative Hierarchical clustering.
    """
    base_pipe = Pipeline([
        ('scaler', RobustScaler()),
        ('model', AgglomerativeClustering(n_clusters=k))
    ])
    
    if for_compare:
        return base_pipe
        
    X = df[predictors]
    base_pipe.fit(X)
    
    print(f"--- Agglomerative Execution ---")
    print(f"Clusters: {k} | Strategy: Bottom-up")
    print("-" * 35)
    
    return base_pipe

#---Function:birch_clustering---
def birch_clustering(df, predictors, k=4, for_compare=False):
    """
    BIRCH clustering - Balanced Iterative Reducing and Clustering using Hierarchies.
    """
    base_pipe = Pipeline([
        ('scaler', RobustScaler()),
        ('model', Birch(n_clusters=k))
    ])
    
    if for_compare:
        return base_pipe
        
    X = df[predictors]
    base_pipe.fit(X)
    
    print(f"--- BIRCH Execution ---")
    print(f"Clusters: {k} | Optimized for: Large Datasets")
    print("-" * 35)
    
    return base_pipe

#---Function:compare_clustering_models---
def compare_clustering_models(df, predictors, k=4):
    """
    Executes and compares all 6 clustering models.
    """
    print("Starting Clustering Models Comparison...")
    print(f"Predictors: {len(predictors)} | Target K: {k}")
    print("-" * 35)

    X = df[predictors]
    results = {}
    
    # Executing each model
    results['Agglomerative'] = agglomerative_clustering(df, predictors, k=k)
    results['BIRCH'] = birch_clustering(df, predictors, k=k)
    results['DBSCAN'] = dbscan_clustering(df, predictors) # Uses internal density
    results['GMM'] = gaussian_mixture_clustering(df, predictors, k=k)
    results['KMeans'] = kmeans_clustering(df, predictors, k=k)
    results['KMedoids'] = kmedoids_clustering(df, predictors, k=k)

    print("\n--- Summary of Clustering Assignments ---")
    summary_data = []
    for name, model in results.items():
        labels = model.fit_predict(X)
        n_clusters = len(np.unique(labels[labels != -1])) # Exclude DBSCAN noise
        df[f'Cluster_{name}'] = labels
        summary_data.append({"Model": name, "Clusters_Detected": n_clusters})

    print(pd.DataFrame(summary_data).to_string(index=False))
    return results

#---Function:dbscan_clustering---
def dbscan_clustering(df, predictors, eps=0.5, min_samples=5, for_compare=False):
    """
    DBSCAN clustering - Density-based spatial clustering.
    """
    base_pipe = Pipeline([
        ('scaler', RobustScaler()),
        ('model', DBSCAN(eps=eps, min_samples=min_samples))
    ])
    
    if for_compare:
        return base_pipe
        
    X = df[predictors]
    base_pipe.fit(X)
    
    print(f"--- DBSCAN Execution ---")
    print(f"Eps: {eps} | Min Samples: {min_samples} | Noise detection: Yes")
    print("-" * 35)
    
    return base_pipe

#---Function:gaussian_mixture_clustering---
def gaussian_mixture_clustering(df, predictors, k=4, for_compare=False):
    """
    Gaussian Mixture Models (GMM) - Expectation-Maximization clustering.
    """
    base_pipe = Pipeline([
        ('scaler', RobustScaler()),
        ('model', GaussianMixture(n_components=k, random_state=42))
    ])
    
    if for_compare:
        return base_pipe
        
    X = df[predictors]
    base_pipe.fit(X)
    
    print(f"--- GMM Execution ---")
    print(f"Components: {k} | Covariance: Full")
    print("-" * 35)
    
    return base_pipe

#---Function:kmeans_clustering---
def kmeans_clustering(df, predictors, k=4, for_compare=False):
    """
    K-Means clustering algorithm.
    """
    base_pipe = Pipeline([
        ('scaler', RobustScaler()),
        ('model', KMeans(n_clusters=k, random_state=42, n_init=10))
    ])
    
    if for_compare:
        return base_pipe
        
    X = df[predictors]
    base_pipe.fit(X)
    
    print(f"--- KMeans Execution ---")
    print(f"Clusters: {k} | Seed: 42")
    print("-" * 35)
    
    return base_pipe

#---Function:kmedoids_clustering---
def kmedoids_clustering(df, predictors, k=4, for_compare=False):
    """
    K-Medoids (PAM) clustering - Uses medoids instead of centroids.
    """
    base_pipe = Pipeline([
        ('scaler', RobustScaler()),
        ('model', KMedoids(n_clusters=k, random_state=42, method='pam'))
    ])
    
    if for_compare:
        return base_pipe
        
    X = df[predictors]
    base_pipe.fit(X)
    
    print(f"--- K-Medoids Execution ---")
    print(f"Clusters: {k} | Robust to Outliers: High")
    print("-" * 35)
    
    return base_pipe
