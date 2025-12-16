import pandas as pd
import numpy as np
from scipy.stats import shapiro, normaltest, anderson, kstest, norm
from scipy.stats import skew, kurtosis

#--- normality_check ---
def normality_check(df, numeric_cols=None):
    """
    Perform multiple normality tests on numeric columns and return a structured DataFrame.
    
    Tests included:
    - Shapiro-Wilk (n <= 5000)
    - D’Agostino K^2 (n > 20)
    - Anderson-Darling (5% significance)
    - Kolmogorov-Smirnov against normal with same mean/std
    
    Parameters:
    - df: pandas DataFrame
    - numeric_cols: list of numeric columns to test (default: all numeric)
    
    Returns:
    - results_df: DataFrame with columns: Column, N, Mean, Std, Shapiro, DAgostino, Anderson, KS
    """
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    results = []
    
    for col in numeric_cols:
        s = df[col].dropna()
        n = len(s)
        col_mean = s.mean()
        col_std = s.std()
        
        #Initialize flags
        shapiro_flag = np.nan
        dagostino_flag = np.nan
        anderson_flag = np.nan
        ks_flag = np.nan
        
        #Shapiro-Wilk
        if n <= 5000:
            _, p = shapiro(s)
            shapiro_flag = p > 0.05
        
        #D’Agostino K^2
        if n > 20:
            _, p = normaltest(s)
            dagostino_flag = p > 0.05
        
        #Anderson-Darling (5%)
        ad_result = anderson(s)
        crit_val_5 = ad_result.critical_values[2]  # 5% significance
        ad_stat = ad_result.statistic
        anderson_flag = ad_stat < crit_val_5
        
        #Kolmogorov-Smirnov against normal
        _, ks_p = kstest(s, 'norm', args=(col_mean, col_std))
        ks_flag = ks_p > 0.05
        
        results.append({
            "Column": col,
            "N": n,
            "Mean": col_mean,
            "Std": col_std,
            "Shapiro-Wilk": shapiro_flag,
            "D’Agostino K²": dagostino_flag,
            "Anderson-Darling (5%)": anderson_flag,
            "Kolmogorov-Smirnov": ks_flag
        })
    
    return pd.DataFrame(results)

#--- Function : numeric_skew_kurt ---
def numeric_skew_kurt(df, numeric_cols):
    """
    Compute skewness and kurtosis for selected numeric columns.

    Parameters:
    - df: pandas DataFrame
    - numeric_cols: list of numeric columns to analyze (must be explicitly provided)

    Returns:
    - skew_kurt_df: DataFrame with skewness and kurtosis for each column
    """
    results = []
    
    for col in numeric_cols:
        if col not in df.columns:
            print(f"Warning: {col} not found in DataFrame. Skipping.")
            continue
            
        col_data = df[col].dropna()
        results.append({
            "Column": col,
            "Skewness": skew(col_data),
            "Kurtosis": kurtosis(col_data)
        })
    
    skew_kurt_df = pd.DataFrame(results)
    return skew_kurt_df
