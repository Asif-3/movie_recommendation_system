import streamlit as st
import pandas as pd
import requests
import os
from datasets import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Page config
st.set_page_config(page_title="🎬 Movie Recommender", layout="wide")

# Define URLs for dataset files on GitHub
arrow_url = 'https://github.com/Asif-3/movie_recommendation_system/raw/main/movies-dataset-train.arrow'
info_url = 'https://github.com/Asif-3/movie_recommendation_system/raw/main/dataset_info.json'

# Directory to save the dataset files
dataset_path = 'datasets'

# Ensure the directory exists
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

# Download Arrow file if not already present
arrow_file_path = os.path.join(dataset_path, 'movies-dataset-train.arrow')
if not os.path.exists(arrow_file_path):
    st.write("Downloading dataset...")
    response = requests.get(arrow_url)
    with open(arrow_file_path, 'wb') as f:
        f.write(response.content)
    st.write("Dataset downloaded!")

# Download dataset_info file if not already present
info_file_path = os.path.join(dataset_path, 'dataset_info.json')
if not os.path.exists(info_file_path):
    response = requests.get(info_url)
    with open(info_file_path, 'wb') as f:
        f.write(response.content)

# Load dataset using Hugging Face's datasets library
@st.cache_data
def load_data():
    dataset = Dataset.load_from_disk(dataset_path)
    df = dataset.to_pandas()

    # Rename columns to lowercase for consistency
    df.rename(columns={
        'Title': 'title',
        'Overview': 'overview',
        'Poster_Url': 'poster_url'
    }, inplace=True)

    # Drop rows with missing title or overview
    df = df.dropna(subset=['title', 'overview']).reset_index(drop=True)
    return df

# Compute similarity
@st.cache_data
def compute_similarity(df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['overview'].fillna(''))
    return cosine_similarity(tfidf_matrix)

# Recommendation logic
def recommend(movie, df, similarity_matrix):
    index = df[df['title'] == movie].index[0]
    distances = list(enumerate(similarity_matrix[index]))
    sorted_distances = sorted(distances, key=lambda x: x[1], reverse=True)[1:6]
    recommendations = [(df.iloc[i].title, df.iloc[i].overview, df.iloc[i].poster_url) for i, _ in sorted_distances]
    return recommendations

# Load data and run app
movies = load_data()

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
    st.warning("Dataset is empty or missing required columns.")
