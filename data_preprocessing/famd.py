from prince import FAMD
import pandas as pd

def perform_famd(X_train, X_test=None, n_components=2, return_model=False):
    """
    Perform Factor Analysis of Mixed Data (FAMD) on mixed categorical and numerical variables.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Raw mixed train data (categorical + numerical, NOT scaled).
    X_test : pd.DataFrame, optional
        Raw mixed test data.
    n_components : int, default=2
        Number of latent dimensions to extract.
    return_model : bool, default=False
        Whether to return the fitted FAMD model.
    """
    
    #Initialize FAMD (Prince library)
    famd = FAMD(n_components=n_components, random_state=42)
    
    #Fit and transform on Train
    X_train_famd = famd.fit_transform(X_train)
    
    #Create descriptive column names
    famd_cols = [f'FAMD_Dim{i+1}' for i in range(n_components)]
    
    #Convert to DataFrame
    df_train_famd = pd.DataFrame(X_train_famd, columns=famd_cols, index=X_train.index)
    
    #Handle Test set if provided
    df_test_famd = None
    if X_test is not None:
        #Align columns to ensure consistent structure
        X_test_aligned = X_test.reindex(columns=X_train.columns)
        X_test_famd = famd.transform(X_test_aligned)
        df_test_famd = pd.DataFrame(X_test_famd, columns=famd_cols, index=X_test.index)
    
    #Return results (flexible return logic)
    outputs = [df_train_famd]
    if df_test_famd is not None:
        outputs.append(df_test_famd)
    
    #Use eigenvalues_summary for explained inertia
    outputs.append(famd.eigenvalues_summary)
    
    if return_model:
        outputs.append(famd)
        
    return tuple(outputs)
