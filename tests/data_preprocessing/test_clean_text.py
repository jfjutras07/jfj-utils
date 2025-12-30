import pandas as pd
from data_preprocessing.text_cleaning import clean_text

#--- Function : test_clean_text_basic ---
def test_clean_text_basic():
    df = pd.DataFrame({
        "comments": ["Hello World!", "This is a test.", None, ""]
    })

    df_clean = clean_text(df)

    assert "comments_clean" in df_clean.columns
    assert all(isinstance(c, str) for c in df_clean["comments_clean"])
    assert df_clean.shape[0] == 2

#--- Function : test_clean_text_lowercase ---
def test_clean_text_lowercase():
    df = pd.DataFrame({
        "comments": ["HELLO WORLD", "Mixed Case Text"]
    })

    df_clean = clean_text(df)

    assert all(c.islower() for c in df_clean["comments_clean"])

#--- Function : test_clean_text_remove_punctuation_numbers ---
def test_clean_text_remove_punctuation_numbers():
    df = pd.DataFrame({
        "comments": ["Hello!!! 123", "Test #hashtag @mention"]
    })

    df_clean = clean_text(df)

    assert all(char.isalpha() or char.isspace() for comment in df_clean["comments_clean"] for char in comment)

#--- Function : test_clean_text_remove_stopwords ---
def test_clean_text_remove_stopwords():
    df = pd.DataFrame({
        "comments": ["this is a test of stopwords removal"]
    })

    df_clean = clean_text(df, remove_stopwords=True, lemmatize=False)

    stop_words = set(["this", "is", "a", "of"])
    words = df_clean["comments_clean"].iloc[0].split()
    assert not any(word in stop_words for word in words)

#--- Function : test_clean_text_lemmatize ---
def test_clean_text_lemmatize():
    df = pd.DataFrame({
        "comments": ["running cats are playing"]
    })

    df_clean = clean_text(df, remove_stopwords=False, lemmatize=True)

    assert "running" not in df_clean["comments_clean"].iloc[0]
    assert "playing" not in df_clean["comments_clean"].iloc[0]
