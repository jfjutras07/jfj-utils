import pandas as pd
import numpy as np

#--- Function : full_eda_summary ---
def full_eda_summary(df, id_cols=None, date_cols=None, cat_threshold=20):
    """
    Generate a comprehensive EDA summary for a pandas DataFrame.

    Parameters:
    - df: pandas DataFrame
    - id_cols: list of column names considered as IDs
    - date_cols: list of column names considered as dates
    - cat_threshold: max unique values for a column to be considered categorical

    Returns:
    - summary: dictionary with keys: 'numeric', 'boolean', 'categorical', 'text', 'id', 'date'
    """
    summary = {}

    #Numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    summary["numeric"] = df[numeric_cols].describe().transpose()

    #Boolean columns
    bool_cols = df.select_dtypes(include=["bool"]).columns
    bool_summary = pd.DataFrame()
    for col in bool_cols:
        bool_summary.loc[col, "True_count"] = df[col].sum()
        bool_summary.loc[col, "False_count"] = (~df[col]).sum()
        bool_summary.loc[col, "Percent_true"] = df[col].mean() * 100
    summary["boolean"] = bool_summary

    #Categorical columns
    cat_cols = [col for col in df.select_dtypes(include=["object", "category"]).columns 
                if df[col].nunique() <= cat_threshold]
    cat_summary = pd.DataFrame()
    for col in cat_cols:
        cat_summary.loc[col, "unique_values"] = df[col].nunique()
        cat_summary.loc[col, "most_frequent"] = df[col].mode()[0] if not df[col].mode().empty else None
        cat_summary.loc[col, "missing_count"] = df[col].isna().sum()
    summary["categorical"] = cat_summary

    #Text / free-form columns
    text_cols = [col for col in df.select_dtypes(include=["object"]).columns if col not in cat_cols]
    text_summary = pd.DataFrame()
    for col in text_cols:
        text_summary.loc[col, "unique_values"] = df[col].nunique()
        text_summary.loc[col, "most_frequent"] = df[col].mode()[0] if not df[col].mode().empty else None
        text_summary.loc[col, "average_length"] = df[col].str.len().mean()
    summary["text"] = text_summary

    #ID columns
    if id_cols is None:
        id_cols = []
    id_summary = pd.DataFrame()
    for col in id_cols:
        if col in df.columns:
            id_summary.loc[col, "unique_count"] = df[col].nunique()
            id_summary.loc[col, "duplicate_count"] = df.shape[0] - df[col].nunique()
    summary["id"] = id_summary

    #Date columns (without coercion)
    if date_cols is None:
        date_cols = []
    date_summary = pd.DataFrame()
    for col in date_cols:
        if col in df.columns:
            series = df[col].astype(str)  # on garde les valeurs originales
            date_summary.loc[col, "unique_count"] = series.nunique()
            date_summary.loc[col, "missing_count"] = series.isna().sum()
            date_summary.loc[col, "example_values"] = ", ".join(series.dropna().unique()[:5])
            # Analyse rapide des séparateurs
            date_summary.loc[col, "dash_count"] = series.str.count("-").sum()
            date_summary.loc[col, "slash_count"] = series.str.count("/").sum()
            date_summary.loc[col, "dot_count"] = series.str.count("\.").sum()
            date_summary.loc[col, "avg_length"] = series.str.len().mean()
    summary["date"] = date_summary

    return summary
