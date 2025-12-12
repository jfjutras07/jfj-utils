import pandas as pd
import numpy as np

#--- Function : describe_structure ---
def describe_structure(df, id_cols=None, date_cols=None, cat_threshold=20):
    """
    Display a clear EDA summary for a pandas DataFrame.
    Prints each section with spacing between them.
    """

    #--- Numeric columns ---
    print("\n=== Numeric Columns ===\n")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print(df[numeric_cols].describe().transpose())

    #--- Boolean columns ---
    print("\n=== Boolean Columns ===\n")
    bool_cols = df.select_dtypes(include=["bool"]).columns
    bool_summary = pd.DataFrame()
    for col in bool_cols:
        bool_summary.loc[col, "True_count"] = df[col].sum()
        bool_summary.loc[col, "False_count"] = (~df[col]).sum()
        bool_summary.loc[col, "Percent_true"] = df[col].mean() * 100
    print(bool_summary)

    #--- Categorical columns ---
    print("\n=== Categorical Columns ===\n")
    cat_cols = [col for col in df.select_dtypes(include=["object","category"]).columns
                if df[col].nunique() <= cat_threshold]
    cat_summary = pd.DataFrame()
    for col in cat_cols:
        cat_summary.loc[col,"unique_values"] = df[col].nunique()
        cat_summary.loc[col,"most_frequent"] = df[col].mode()[0] if not df[col].mode().empty else None
        cat_summary.loc[col,"missing_count"] = df[col].isna().sum()
    print(cat_summary)

    #--- Text columns ---
    print("\n=== Text Columns ===\n")
    text_cols = [col for col in df.select_dtypes(include=["object"]).columns if col not in cat_cols]
    text_summary = pd.DataFrame()
    for col in text_cols:
        text_summary.loc[col,"unique_values"] = df[col].nunique()
        text_summary.loc[col,"most_frequent"] = df[col].mode()[0] if not df[col].mode().empty else None
        text_summary.loc[col,"average_length"] = df[col].str.len().mean()
    print(text_summary)

    #--- ID columns ---
    print("\n=== ID Columns ===\n")
    if id_cols is None:
        id_cols = []
    id_summary = pd.DataFrame()
    for col in id_cols:
        if col in df.columns:
            id_summary.loc[col,"unique_count"] = df[col].nunique()
            id_summary.loc[col,"duplicate_count"] = df.shape[0] - df[col].nunique()
    print(id_summary)

    #--- Date columns ---
    print("\n=== Date Columns ===\n")
    if date_cols is None:
        date_cols = []
    date_summary = pd.DataFrame()
    for col in date_cols:
        if col in df.columns:
            series = df[col].astype(str)
            date_summary.loc[col,"unique_count"] = series.nunique()
            date_summary.loc[col,"missing_count"] = series.isna().sum()
            date_summary.loc[col,"example_values"] = ", ".join(series.dropna().unique()[:5])
            date_summary.loc[col,"dash_count"] = series.str.count("-").sum()
            date_summary.loc[col,"slash_count"] = series.str.count("/").sum()
            date_summary.loc[col,"dot_count"] = series.str.count("\.").sum()
            date_summary.loc[col,"avg_length"] = series.str.len().mean()
    print(date_summary)
