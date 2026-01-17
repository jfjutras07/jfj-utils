import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Optional
import re
from typing import Dict
import string

#--- Class : column_dropper ---
class column_dropper(BaseEstimator, TransformerMixin):
    """
    Drops specified columns and can automatically remove constant columns 
    (zero variance) detected during the fit process.
    """
    def __init__(self, columns: Optional[List[str]] = None, drop_constant: bool = False):
        self.columns = columns if columns else []
        self.drop_first = drop_constant
        self.constant_cols_ = []

    def fit(self, X: pd.DataFrame, y=None):
        # Automatically identify columns with only 1 unique value
        if self.drop_first:
            self.constant_cols_ = [col for col in X.columns if X[col].nunique() <= 1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        # Combine manual list and detected constant columns
        total_to_drop = list(set(self.columns + self.constant_cols_))
        
        # Drop only if columns exist in the current dataframe
        existing_cols = [c for c in total_to_drop if c in X.columns]
        return X.drop(columns=existing_cols)

#--- Function: clean_names ---
def clean_names(df: pd.DataFrame, first_col: str='first_name', last_col: str='last_name') -> pd.DataFrame:
    """
    Clean first and last name columns in a DataFrame.
    Steps:
    - Strip whitespace and standardize missing values
    - Proper capitalization (hyphens, apostrophes, Mc/Mac)
    - Split multi-part first names if needed
    - Merge extracted last name with original last name
    """
    df[first_col] = df[first_col].astype(str).str.strip().replace(['nan','None','', 'NaN','<na>'], pd.NA)
    df[last_col] = df[last_col].astype(str).str.strip().replace(['nan','None','', 'NaN','<na>'], pd.NA)

    def proper_case(name: str) -> str:
        if pd.isna(name) or str(name).strip() == '':
            return pd.NA
        name = str(name).lower()
        name = re.sub(r"(^|[-'])\w", lambda m: m.group().upper(), name)
        name = re.sub(r'\b(Mc)(\w)', lambda m: m.group(1) + m.group(2).upper(), name)
        name = re.sub(r'\b(Mac)(\w)', lambda m: m.group(1) + m.group(2).upper(), name)
        return name

    df[first_col] = df[first_col].apply(proper_case)
    df[last_col] = df[last_col].apply(proper_case)

    split_names = df[first_col].str.split(' ', n=1, expand=True)
    df['first_name_clean'] = split_names[0]
    df['last_extracted'] = split_names[1] if split_names.shape[1] > 1 else pd.NA

    df['last_name_clean'] = df.apply(
        lambda row: row[last_col] if pd.notna(row[last_col]) and str(row[last_col]).strip()!=str(row['last_extracted']).strip() else row['last_extracted'],
        axis=1
    )

    df['last_name_clean'] = df['last_name_clean'].apply(lambda x: pd.NA if pd.isna(x) or str(x).strip()=='' else x)
    df.drop(columns=['last_extracted'], inplace=True)

    return df

#--- Function: clean_names_multiple ---
def clean_names_multiple(dfs: Dict[str,pd.DataFrame], first_col: str='first_name', last_col: str='last_name') -> Dict[str,pd.DataFrame]:
    """
    Apply clean_names to multiple DataFrames in a dictionary.
    """
    for key, df in dfs.items():
        df_clean = clean_names(df, first_col=first_col, last_col=last_col)
        df_clean.drop(columns=[first_col,last_col], inplace=True)
        df_clean.rename(columns={'first_name_clean':'first_name','last_name_clean':'last_name'}, inplace=True)
        dfs[key] = df_clean
    return dfs

#--- Function : clean_text ---
def clean_text(df, text_col='comments', new_col='comments_clean', 
               remove_stopwords=True, lemmatize=True):
    """
    Generic text cleaning and preprocessing function.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the text column.
    text_col : str
        Name of the raw text column.
    new_col : str
        Name of the column to store cleaned text.
    remove_stopwords : bool
        Remove stopwords if True.
    lemmatize : bool
        Apply lemmatization if True.
    
    Returns:
    --------
    df : pd.DataFrame
        DataFrame with the new cleaned text column.
    """
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    nltk.download('stopwords')
    nltk.download('wordnet')
    
    #Drop missing or empty comments
    df = df.dropna(subset=[text_col]).copy()
    df[text_col] = df[text_col].astype(str).str.strip()
    df = df[df[text_col] != ''].copy()
    
    #Copy text to new column
    df[new_col] = df[text_col].copy()
    
    #Lowercase text
    df[new_col] = df[new_col].str.lower()
    
    #Remove URLs, emails, mentions, hashtags
    df[new_col] = df[new_col].apply(lambda x: re.sub(r'http\S+|www\S+|https\S+', '', x))
    df[new_col] = df[new_col].apply(lambda x: re.sub(r'\S+@\S+', '', x))
    df[new_col] = df[new_col].apply(lambda x: re.sub(r'@\w+', '', x))
    df[new_col] = df[new_col].apply(lambda x: re.sub(r'#\w+', '', x))
    
    #Remove punctuation and numbers
    df[new_col] = df[new_col].apply(lambda x: re.sub(f'[{re.escape(string.punctuation)}0-9]', ' ', x))
    
    #Remove extra spaces
    df[new_col] = df[new_col].apply(lambda x: re.sub(r'\s+', ' ', x).strip())
    
    #Remove stopwords
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        df[new_col] = df[new_col].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
    
    #Lemmatize words
    if lemmatize:
        lemmatizer = WordNetLemmatizer()
        df[new_col] = df[new_col].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))
    
    return df
