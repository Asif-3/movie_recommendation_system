import streamlit as st
import pandas as pd
import base64
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="🎬 Movie Recommender", layout="wide")

def add_bg_from_local(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: none;
        }}
        .stApp::before {{
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            filter: blur(6px) brightness(0.5);
            z-index: -1;
        }}

        /* Custom Red Button */
        div.stButton > button:first-child {{
            background-color: #ff4b4b;
            color: white;
            font-weight: bold;
            border-radius: 8px;
            padding: 0.6em 1.2em;
            border: none;
        }}
        div.stButton > button:first-child:hover {{
            background-color: #e63939;
            color: white;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_from_local("as.png")


@st.cache_data
def load_data():
    df = pd.read_csv("dataset.csv")
    df.rename(columns={'Title':'title','Overview':'overview','Poster_Url':'poster_url'}, inplace=True)
    df = df.dropna(subset=['title','overview']).reset_index(drop=True)
    return df

@st.cache_data
def compute_similarity(df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['overview'].fillna(''))
    return cosine_similarity(tfidf_matrix)


def recommend(movie, df, similarity_matrix):
    matches = df[df['title'].str.strip().str.lower() == movie.strip().lower()]
    if matches.empty:
        return []
    index = matches.index[0]
    distances = list(enumerate(similarity_matrix[index]))
    sorted_distances = sorted(distances, key=lambda x: x[1], reverse=True)[1:6]
    return [(df.iloc[i].title, df.iloc[i].overview, df.iloc[i].poster_url) for i,_ in sorted_distances]

movies = load_data()

if not movies.empty:
    similarity = compute_similarity(movies)

    st.title("🎬 Movie Recommender System")
    selected_movie = st.selectbox("🎞️ Select a movie", movies['title'].values)

    if st.button("✨ Show Recommendations"):
        st.subheader("You may also like:")
        results = recommend(selected_movie, movies, similarity)

        if not results:
            st.warning("⚠️ No recommendations found for this movie.")
        else:
            cols = st.columns(len(results))
            for i,(title, overview, poster_url) in enumerate(results):
                with cols[i]:
                    st.image(
                        poster_url if pd.notna(poster_url) else "https://via.placeholder.com/200x300?text=No+Image",
                        width=150
                    )
                    st.markdown(f"**🎥 {title}**")
                    st.caption(overview[:150] + "...")
else:
    st.warning("Dataset is empty or missing required columns.")
