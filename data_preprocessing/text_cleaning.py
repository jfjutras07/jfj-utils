import re
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

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
