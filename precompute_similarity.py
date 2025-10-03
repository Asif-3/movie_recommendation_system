import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load your dataset
df = pd.read_csv("dataset.csv")
df.rename(columns={'Title':'title','Overview':'overview','Poster_Url':'poster_url'}, inplace=True)
df = df.dropna(subset=['title','overview']).reset_index(drop=True)

# Compute TF-IDF matrix and cosine similarity
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['overview'].fillna(''))
similarity = cosine_similarity(tfidf_matrix)

# Save similarity matrix and titles
np.save("similarity_matrix.npy", similarity)
df['title'].to_csv("movie_titles.csv", index=False)
