import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Page config
st.set_page_config(page_title="🎬 Movie Recommender", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
    /* Container cards for recommendations */
    .recommendation-card {
        background-color: #f9f9f9;
        padding: 15px;
        border-radius: 15px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        text-align: center;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        animation: fadeInUp 0.5s ease forwards;
        margin-top: 10px;
    }

    .recommendation-card:hover {
        transform: scale(1.05);
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }

    .recommendation-card img {
        border-radius: 10px;
        max-height: 220px;
        margin-bottom: 10px;
        transition: transform 0.3s ease;
    }

    .recommendation-card h4 {
        margin: 5px 0;
        font-size: 18px;
    }

    .recommendation-card p {
        font-size: 14px;
        color: #555;
    }

    @keyframes fadeInUp {
        0% {
            opacity: 0;
            transform: translateY(20px);
        }
        100% {
            opacity: 1;
            transform: translateY(0);
        }
    }
    </style>
""", unsafe_allow_html=True)

# Load dataset from CSV
@st.cache_resource
def load_data():
    try:
        df = pd.read_csv("dataset.csv")
        df.rename(columns={
            'Title': 'title',
            'Overview': 'overview',
            'Poster_Url': 'poster_url'
        }, inplace=True)
        df = df.dropna(subset=['title', 'overview']).reset_index(drop=True)
        return df
    except FileNotFoundError:
        st.error("❌ dataset.csv not found. Please upload it alongside your app file.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Error loading dataset: {e}")
        return pd.DataFrame()

# Compute similarity matrix
@st.cache_resource
def compute_similarity(df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['overview'].fillna(''))
    return cosine_similarity(tfidf_matrix)

# Recommend similar movies
def recommend(movie, df, similarity_matrix):
    try:
        index = df[df['title'] == movie].index[0]
    except IndexError:
        st.error("❌ Movie not found in dataset.")
        return []

    distances = list(enumerate(similarity_matrix[index]))
    sorted_distances = sorted(distances, key=lambda x: x[1], reverse=True)[1:6]
    recommendations = [(df.iloc[i].title, df.iloc[i].overview, df.iloc[i].poster_url) for i, _ in sorted_distances]
    return recommendations

# Initialize session state
if 'show' not in st.session_state:
    st.session_state['show'] = False

# Load data and run the app
movies = load_data()

if not movies.empty and 'title' in movies.columns:
    try:
        similarity = compute_similarity(movies)

        st.title("🎬 Movie recommender system - Created by Asif")
        selected_movie = st.selectbox("🎞️ Select a movie", movies['title'].values)

        if st.button("✨ Show Recommendations"):
            st.session_state['show'] = True

        if st.session_state['show']:
            if selected_movie:
                st.subheader("You may also like:")
                results = recommend(selected_movie, movies, similarity)

                if results:
                    cols = st.columns(len(results))
                    for i, (title, overview, poster_url) in enumerate(results):
                        with cols[i]:
                            st.markdown(f"""
                                <div class="recommendation-card">
                                    <img src="{poster_url if pd.notna(poster_url) else 'https://via.placeholder.com/200x300?text=No+Image'}" width="150">
                                    <h4>{title}</h4>
                                    <p>{overview[:150]}...</p>
                                </div>
                            """, unsafe_allow_html=True)
                else:
                    st.warning("⚠️ No recommendations found.")
            else:
                st.warning("⚠️ Please select a movie.")
    except Exception as e:
        st.error(f"💥 Something went wrong: {e}")
else:
    st.warning("⚠️ Dataset is empty or missing required columns.")