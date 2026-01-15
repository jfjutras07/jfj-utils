import pandas as pd
import numpy as np

#--- Function : describe_structure ---
def describe_structure(df, id_cols=None, date_cols=None, cat_threshold=20, max_unique_display=50):
    """
    Display a clear EDA summary for a pandas DataFrame.
    Strict exclusion logic ensures columns don't repeat across categories.
    """
    import pandas as pd
    import numpy as np

    # Normalize inputs to lists
    if id_cols is None: id_cols = []
    if isinstance(id_cols, str): id_cols = [id_cols]
    if date_cols is None: date_cols = []
    if isinstance(date_cols, str): date_cols = [date_cols]

    # Automatic detection of datetime types
    auto_date_cols = df.select_dtypes(include=['datetime', 'datetimetz']).columns.tolist()
    all_date_cols = list(set(date_cols + auto_date_cols))
    
    # Identify boolean columns
    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()

    # Define exclusion list to ensure mutual exclusivity
    # Columns in this list will not appear in Numeric or Categorical sections
    excluded_from_stats = id_cols + all_date_cols + bool_cols

    # --- Numeric columns ---
    print("\n=== Numeric Columns ===\n")
    all_numeric = df.select_dtypes(include=[np.number]).columns
    actual_numeric = [c for c in all_numeric if c not in excluded_from_stats]
    if actual_numeric:
        print(df[actual_numeric].describe().transpose())
    else:
        print("Empty DataFrame\nColumns: []\nIndex: []")

    # --- Boolean columns ---
    print("\n=== Boolean Columns ===\n")
    if bool_cols:
        bool_summary = pd.DataFrame()
        for col in bool_cols:
            bool_summary.loc[col, "True_count"] = df[col].sum()
            bool_summary.loc[col, "False_count"] = (~df[col]).sum()
            bool_summary.loc[col, "Percent_true"] = df[col].mean() * 100
        print(bool_summary)
    else:
        print("Empty DataFrame\nColumns: []\nIndex: []")

    # --- Categorical columns ---
    print("\n=== Categorical Columns ===\n")
    # Candidates are objects/categories not already classified as IDs or Dates
    all_cat_candidates = df.select_dtypes(include=["object", "category"]).columns
    actual_cat_candidates = [c for c in all_cat_candidates if c not in excluded_from_stats]
    
    cat_cols = [col for col in actual_cat_candidates if df[col].nunique() <= cat_threshold]
    if cat_cols:
        cat_summary = pd.DataFrame()
        for col in cat_cols:
            cat_summary.loc[col, "unique_values"] = df[col].nunique()
            cat_summary.loc[col, "most_frequent"] = df[col].mode()[0] if not df[col].mode().empty else None
            
            unique_vals = df[col].dropna().unique()
            if len(unique_vals) > max_unique_display:
                unique_vals = list(unique_vals[:max_unique_display]) + ["..."]
            cat_summary.loc[col, "all_unique_values"] = ", ".join(map(str, unique_vals))
            cat_summary.loc[col, "missing_count"] = df[col].isna().sum()
        print(cat_summary)
    else:
        print("Empty DataFrame\nColumns: []\nIndex: []")

    # --- Text columns ---
    print("\n=== Text Columns ===\n")
    text_cols = [col for col in actual_cat_candidates if col not in cat_cols]
    if text_cols:
        text_summary = pd.DataFrame()
        for col in text_cols:
            text_summary.loc[col, "unique_values"] = df[col].nunique()
            text_summary.loc[col, "most_frequent"] = df[col].mode()[0] if not df[col].mode().empty else None
            text_summary.loc[col, "average_length"] = df[col].str.len().mean()
        print(text_summary)
    else:
        print("Empty DataFrame\nColumns: []\nIndex: []")

    # --- ID columns ---
    print("\n=== ID Columns ===\n")
    if id_cols:
        id_summary = pd.DataFrame()
        for col in id_cols:
            if col in df.columns:
                id_summary.loc[col, "unique_count"] = df[col].nunique()
                id_summary.loc[col, "duplicate_count"] = df.shape[0] - df[col].nunique()
        print(id_summary)
    else:
        print("Empty DataFrame\nColumns: []\nIndex: []")

    # --- Date columns ---
    print("\n=== Date Columns ===\n")
    if all_date_cols:
        date_summary = pd.DataFrame()
        for col in all_date_cols:
            if col in df.columns:
                series = df[col].astype(str)
                date_summary.loc[col, "unique_count"] = series.nunique()
                date_summary.loc[col, "missing_count"] = df[col].isna().sum()
                date_summary.loc[col, "example_values"] = ", ".join(series.dropna().unique()[:5])
                date_summary.loc[col, "dash_count"] = series.str.count("-").sum()
                date_summary.loc[col, "slash_count"] = series.str.count("/").sum()
                date_summary.loc[col, "dot_count"] = series.str.count(r"\.").sum()
                date_summary.loc[col, "avg_length"] = series.str.len().mean()
        print(date_summary)
    else:
        print("Empty DataFrame\nColumns: []\nIndex: []")
