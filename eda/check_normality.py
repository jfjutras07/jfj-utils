import pandas as pd
import numpy as np
from scipy.stats import shapiro, normaltest, anderson, kstest, norm

#--- Function : normality_check ---
def normality_check(df, numeric_cols=None):
    """
    Perform normality tests automatically based on sample size, including Shapiro-Wilk, D’Agostino K^2, 
    Anderson-Darling, and Kolmogorov-Smirnov.
    
    Parameters:
    - df: pandas DataFrame
    - numeric_cols: list of numeric columns to test (default: all numeric)
    
    Returns:
    - results_df: DataFrame with test results for each column
    """
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    results = []
    
    for col in numeric_cols:
        s = df[col].dropna()
        n = len(s)
        col_mean = s.mean()
        col_std = s.std()
        
        tests_applied = []
        p_values = []
        normal_flags = []
        
        #Shapiro-Wilk for small/medium sample
        if n <= 5000:
            stat, p = shapiro(s)
            tests_applied.append("Shapiro-Wilk")
            p_values.append(p)
            normal_flags.append(p > 0.05)
        
        #D’Agostino K^2 for n > 20
        if n > 20:
            stat, p = normaltest(s)
            tests_applied.append("D’Agostino K^2")
            p_values.append(p)
            normal_flags.append(p > 0.05)
        
        #Anderson-Darling (always)
        ad_result = anderson(s)
        crit_val_5 = ad_result.critical_values[2]  # 5% significance
        ad_stat = ad_result.statistic
        is_normal_ad = ad_stat < crit_val_5
        tests_applied.append("Anderson-Darling (5%)")
        p_values.append(np.nan)
        normal_flags.append(is_normal_ad)
        
        #Kolmogorov-Smirnov against normal with same mean/std
        ks_stat, ks_p = kstest(s, 'norm', args=(col_mean, col_std))
        tests_applied.append("Kolmogorov-Smirnov")
        p_values.append(ks_p)
        normal_flags.append(ks_p > 0.05)
        
        results.append({
            "Column": col,
            "N": n,
            "Mean": col_mean,
            "Std": col_std,
            "Tests": ", ".join(tests_applied),
            "P-values": ", ".join([f"{pv:.4f}" if not pd.isna(pv) else "NA" for pv in p_values]),
            "Normality_flags": ", ".join([str(flag) for flag in normal_flags])
        })
    
    return pd.DataFrame(results)
