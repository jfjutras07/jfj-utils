from prince import MCA
import pandas as pd

def perform_mca(X_train, X_test=None, n_components=2, return_model=False):
    """
    Perform Multiple Correspondence Analysis (MCA) on categorical/ordinal groups.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Raw categorical/ordinal train data (NOT scaled).
    X_test : pd.DataFrame, optional
        Raw categorical/ordinal test data.
    n_components : int, default=2
        Number of latent dimensions to extract.
    return_model : bool, default=False
        Whether to return the fitted MCA model.
    """
    
    #Initialize MCA (Prince library)
    mca = MCA(n_components=n_components, random_state=42)
    
    #Fit and transform on Train
    X_train_mca = mca.fit_transform(X_train)
    
    #Create descriptive column names
    mca_cols = [f'MCA_Dim{i+1}' for i in range(n_components)]
    
    #Convert to DataFrame
    df_train_mca = pd.DataFrame(X_train_mca, columns=mca_cols, index=X_train.index)
    
    #Handle Test set if provided
    df_test_mca = None
    if X_test is not None:
        # Align columns to ensure the same categories are present
        X_test_aligned = X_test.reindex(columns=X_train.columns)
        X_test_mca = mca.transform(X_test_aligned)
        df_test_mca = pd.DataFrame(X_test_mca, columns=mca_cols, index=X_test.index)
    
    #Return results (flexible return logic)
    outputs = [df_train_mca]
    if df_test_mca is not None:
        outputs.append(df_test_mca)
    
    #Use explained_inertia_ (MCA version of explained_variance_ratio_)
    outputs.append(mca.eigenvalues_summary) 
    
    if return_model:
        outputs.append(mca)
        
    return tuple(outputs)
