import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(page_title="🎬 Movie Recommender", layout="wide")


def hide_github_icon_and_footer():
    hide_streamlit_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .css-164nlkn.egzxvld1 {display: none;}  /* GitHub icon in top right */
        </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Call the function at the top of your app
hide_github_icon_and_footer()


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

@st.cache_resource
def compute_similarity(df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['overview'].fillna(''))
    return cosine_similarity(tfidf_matrix)

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

if 'show' not in st.session_state:
    st.session_state['show'] = False


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
                            st.image(
                                poster_url if pd.notna(poster_url) else "https://via.placeholder.com/200x300?text=No+Image",
                                width=150
                            )
                            st.markdown(f"**🎥 {title}**")
                            st.caption(overview[:150] + "...")
                else:
                    st.warning("⚠️ No recommendations found.")
            else:
                st.warning("⚠️ Please select a movie.")
    except Exception as e:
        st.error(f"💥 Something went wrong: {e}")
else:
    st.warning("⚠️ Dataset is empty or missing required columns.")
