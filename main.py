import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Page config
st.set_page_config(page_title="🎬 Movie Recommender", layout="wide")

# Load dataset from CSV
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("dataset.csv")

        # Rename columns to lowercase for consistency
        df.rename(columns={
            'Title': 'title',
            'Overview': 'overview',
            'Poster_Url': 'poster_url'
        }, inplace=True)

        # Drop rows with missing title or overview
        df = df.dropna(subset=['title', 'overview']).reset_index(drop=True)
        return df
    except FileNotFoundError:
        st.error("❌ dataset.csv not found. Please upload it alongside your app file.")
        return pd.DataFrame()

# Compute similarity matrix
@st.cache_data
def compute_similarity(df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['overview'].fillna(''))
    return cosine_similarity(tfidf_matrix)

# Recommend similar movies
def recommend(movie, df, similarity_matrix):
    try:
        index = df[df['title'] == movie].index[0]
    except IndexError:
        return []

    distances = list(enumerate(similarity_matrix[index]))
    sorted_distances = sorted(distances, key=lambda x: x[1], reverse=True)[1:6]
    recommendations = [(df.iloc[i].title, df.iloc[i].overview, df.iloc[i].poster_url) for i, _ in sorted_distances]
    return recommendations

# Load data and run the app
movies = load_data()

if not movies.empty:
    similarity = compute_similarity(movies)

    st.title("🎬 Movie Recommender System")
    selected_movie = st.selectbox("🎞️ Select a movie", movies['title'].values)

    if st.button("✨ Show Recommendations"):
        st.subheader("You may also like:")
        results = recommend(selected_movie, movies, similarity)

        if results:
            cols = st.columns(len(results))
            for i, (title, overview, poster_url) in enumerate(results):
                with cols[i]:
                    st.image(
                        poster_url if pd.notna(poster_url) else "https://via.placeholder.com/200x300?text=No+Image",
                        width=150
                    )
                    st.markdown(f"**🎥 {title}**")
                    st.caption(overview[:150] + "...")
        else:
            st.warning("⚠️ No recommendations found.")
else:
    st.warning("⚠️ Dataset is empty or missing required columns.")
