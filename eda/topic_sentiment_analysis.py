import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon')

##--- Function : topic_sentiment_analysis ---
def topic_sentiment_analysis(df, text_col='comments_clean', n_topics=5):
    """
    Perform topic modeling (LDA) and sentiment analysis on a text column.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the cleaned text.
    text_col : str
        Column name containing the cleaned text.
    n_topics : int
        Number of LDA topics to extract.
    
    Returns:
    --------
    df_out : pd.DataFrame
        Original DataFrame with added columns:
        - 'dominant_topic': assigned topic number (1 to n_topics)
        - 'sentiment_score': VADER compound sentiment score
        - 'sentiment_label': 'positive', 'neutral', or 'negative'
    topic_words : dict
        Dictionary of top 10 words per topic.
    """
    
    #TF-IDF vectorization
    tfidf_vectorizer = TfidfVectorizer(max_df=0.9, min_df=2)
    tfidf_matrix = tfidf_vectorizer.fit_transform(df[text_col])
    
    #LDA topic modeling
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(tfidf_matrix)
    
    #Assign dominant topic to each comment
    topic_distribution = lda.transform(tfidf_matrix)
    df['dominant_topic'] = np.argmax(topic_distribution, axis=1) + 1  # topics numbered 1..n_topics
    
    #Extract top 10 words per topic
    feature_names = tfidf_vectorizer.get_feature_names_out()
    topic_words = {}
    for idx, topic in enumerate(lda.components_):
        top_words = [feature_names[i] for i in topic.argsort()[-10:][::-1]]
        topic_words[idx+1] = top_words  # topics 1..n_topics
    
    #Sentiment analysis using VADER
    sia = SentimentIntensityAnalyzer()
    df['sentiment_score'] = df[text_col].apply(lambda x: sia.polarity_scores(x)['compound'])
    
    #Label sentiment
    def label_sentiment(score):
        if score >= 0.05:
            return 'positive'
        elif score <= -0.05:
            return 'negative'
        else:
            return 'neutral'
        
    df['sentiment_label'] = df['sentiment_score'].apply(label_sentiment)
    
    return df, topic_words
