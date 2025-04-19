import streamlit as st
import pandas as pd
import requests
import os
from datasets import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Page config
st.set_page_config(page_title="🎬 Movie Recommender", layout="wide")

# GitHub URL for the dataset
arrow_url = 'https://github.com/Asif-3/movie_recommendation_system/raw/main/movies-dataset-train.arrow'

# Local path to store the dataset
dataset_path = 'datasets'
arrow_file_path = os.path.join(dataset_path, 'movies-dataset-train.arrow')

# Create dataset directory if it doesn't exist
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

# Download the .arrow file if not already present
if not os.path.exists(arrow_file_path):
    st.info("📥 Downloading dataset...")
    response = requests.get(arrow_url)
    with open(arrow_file_path, 'wb') as f:
        f.write(response.content)
    st.success("✅ Dataset downloaded!")

# Load dataset using Hugging Face's `from_file` method
@st.cache_data
def load_data():
    dataset = Dataset.from_file(arrow_file_path)
    df = dataset.to_pandas()

    # Standardize column names
    df.rename(columns={
        'Title': 'title',
        'Overview': 'overview',
        'Poster_Url': 'poster_url'
    }, inplace=True)

    # Clean up missing data
    df = df.dropna(subset=['title', 'overview']).reset_index(drop=True)
    return df

# Compute similarity matrix
@st.cache_data
def compute_similarity(df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['overview'].fillna(''))
    return cosine_similarity(tfidf_matrix)

# Recommender function
def recommend(movie, df, similarity_matrix):
    index = df[df['title'] == movie].index[0]
    distances = list(enumerate(similarity_matrix[index]))
    sorted_distances = sorted(distances, key=lambda x: x[1], reverse=True)[1:6]
    recommendations = [(df.iloc[i].title, df.iloc[i].overview, df.iloc[i].poster_url) for i, _ in sorted_distances]
    return recommendations

# Load data
movies = load_data()

# App UI
if not movies.empty:
    similarity = compute_similarity(movies)

    st.title("🎬 Movie Recommender System")
    selected_movie = st.selectbox("🎞️ Select a movie", movies['title'].values)

    if st.button("✨ Show Recommendations"):
        st.subheader("You may also like:")
        results = recommend(selected_movie, movies, similarity)

        cols = st.columns(5)
        for i, (title, overview, poster_url) in enumerate(results):
            with cols[i]:
                st.image(poster_url if pd.notna(poster_url) else "https://via.placeholder.com/200x300?text=No+Image", width=150)
                st.markdown(f"**🎥 {title}**")
                st.caption(overview[:150] + "...")
else:
    st.warning("⚠️ Dataset is empty or missing required columns.")
