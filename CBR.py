import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re


def load_dataset(file_path):
    """
    Load dataset from a CSV file.
    Args:
        file_path (str): Path to the CSV file.
    Returns:
        pd.DataFrame: Loaded dataset.
    """
    df = pd.read_csv(file_path, encoding='Latin1')
    return df

def combine_features(df):
    """
    Combine description and cast columns into one for text similarity.
    Args:
        df (pd.DataFrame): Dataset with 'description' and 'cast' columns.
    Returns:
        pd.DataFrame: Dataset with a new 'combined_features' column.
    """
    df['combined_features'] = df['overview'].fillna('') + ' ; ' + df['cast'].fillna('') + ' ; ' + df['genres'].fillna('') + ' ; ' + df['keywords'].fillna('') + ' ; ' + df['tagline'].fillna('')
    return df

# Preprocess text data (lowercase, remove punctuation)
def preprocess_text(text):
    """
    Clean the text by lowering case and removing special characters.
    Args:
        text (str): Input text string.
    Returns:
        str: Cleaned text string.
    """
    text = text.lower()  # Lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
    return text


def preprocess_dataset(df):
    """
    Apply preprocessing to the combined features.
    Args:
        df (pd.DataFrame): Dataset with combined features.
    Returns:
        pd.DataFrame: Dataset with cleaned combined features.
    """
    df['combined_features'] = df['combined_features'].apply(preprocess_text)
    return df

# Main function to load and preprocess the data
def main(file_path):
    df = load_dataset(file_path)
    df = combine_features(df)
    df = preprocess_dataset(df)
    # print("Dataset loaded and preprocessed successfully!")
    # print(df[['original_title', 'combined_features']].head())
    return df

file_path = 'tmdb-movies500.csv'  # Replace with your dataset path
dataset = main(file_path)

#  Vectorize text using TF-IDF
def vectorize_text(df):
    """
    Transform the combined_features column into TF-IDF vectors.
    Args:
        df (pd.DataFrame): Dataset with combined_features.
    Returns:
        TfidfVectorizer, TF-IDF matrix
    """
    tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), min_df=2)
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])
    return tfidf, tfidf_matrix

#  Compute recommendations
def recommend_items(user_input, tfidf, tfidf_matrix, df, top_n=5):
    """
    Recommend top N items based on user input using cosine similarity.
    Args:
        user_input (str): User's preference description.
        tfidf (TfidfVectorizer): Fitted TF-IDF vectorizer.
        tfidf_matrix (sparse matrix): TF-IDF matrix of dataset.
        top_n (int): Number of recommendations to return.
    Returns:
        pd.DataFrame: Top N recommended items with similarity scores.
    """
    user_vector = tfidf.transform([user_input])
    similarity_scores = cosine_similarity(user_vector, tfidf_matrix).flatten()
    top_indices = similarity_scores.argsort()[-top_n:][::-1]
    recommendations = df.iloc[top_indices][['original_title', 'release_date', 'combined_features']].copy()
    recommendations['similarity_score'] = similarity_scores[top_indices]
    return recommendations

# Main function to recommend based on user input
def main_recommendation(user_query, dataset):
    tfidf, tfidf_matrix = vectorize_text(dataset)
    recommendations = recommend_items(user_query, tfidf, tfidf_matrix, dataset, top_n=5)
    print("\nTop Recommendations:")
    for index, row in recommendations.iterrows():
        print(f"{row['original_title']} ({row['release_date']}) - {row['similarity_score']*100:.2f}% Match")
    return recommendations

def test(ip):
    recommendations = main_recommendation(ip, dataset)

if __name__ == "__main__":
    recommend_input = input("What do you feel like watching today ??")
    main_recommendation = test(recommend_input)
    