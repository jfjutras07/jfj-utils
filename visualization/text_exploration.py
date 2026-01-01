import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer

#--- Function : text_exploration ---
def text_exploration(df, text_col='comments_clean', n_top_words=20):
    """
    Generic function for basic text exploration:
    - Word Cloud
    - Most frequent words bar plot
    - Comment length distribution
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the cleaned text column.
    text_col : str
        Column name with cleaned text.
    n_top_words : int
        Number of top words to display in bar plot.
    """
    
    #Combine all text
    all_text = ' '.join(df[text_col])
    
    #Word Cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
    plt.figure(figsize=(12,6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud')
    plt.show()
    
    #Most frequent words
    vectorizer = CountVectorizer(max_df=0.9, min_df=2)
    word_count = vectorizer.fit_transform(df[text_col])
    words = vectorizer.get_feature_names_out()
    counts = word_count.toarray().sum(axis=0)
    
    freq_df = pd.DataFrame({'word': words, 'count': counts}).sort_values(by='count', ascending=False)
    plt.figure(figsize=(10,6))
    sns.barplot(x='count', y='word', data=freq_df.head(n_top_words), palette='magma')
    plt.title(f'Top {n_top_words} Most Frequent Words')
    plt.show()
    
    #Comment length distribution
    df['comment_length'] = df[text_col].apply(lambda x: len(x.split()))
    plt.figure(figsize=(10,5))
    sns.histplot(df['comment_length'], bins=20, kde=True)
    plt.title('Comment Length Distribution')
    plt.xlabel('Number of Words')
    plt.ylabel('Frequency')
    plt.show()
