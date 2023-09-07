import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import requests  # Add this line to import the 'requests' library

# Load the datasets
data_credit = pd.read_csv('tmdb_5000_credits.csv')
data_movies = pd.read_csv('tmdb_5000_movies.csv')

# Create a TF-IDF vectorizer to convert text data into numerical features
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# Combine the movie's genres and overview into a single text column
data_movies['genres'] = data_movies['genres'].apply(lambda x: ' '.join([genre['name'] for genre in eval(x)]))
data_movies['overview'] = data_movies['overview'].fillna('')
data_movies['content'] = data_movies['genres'] + ' ' + data_movies['overview']

# Fit and transform the TF-IDF vectorizer on the 'content' column
tfidf_matrix = tfidf_vectorizer.fit_transform(data_movies['content'])

# Calculate the cosine similarity between movies
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Create a mapping of movie titles to their respective indices
indices = pd.Series(data_movies.index, index=data_movies['title'])


def fetch_poster(movie_id):
    url = "https://api.themoviedb.org/3/movie/{}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US".format(movie_id)
    data = requests.get(url)
    data = data.json()
    poster_path = data['poster_path']
    full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
    return full_path

# Function to get movie recommendations based on title


def get_recommendations(title, cosine_sim=cosine_sim):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # Get the top 10 most similar movies
    movie_indices = [i[0] for i in sim_scores]
    recommended_movie_names = data_movies['title'].iloc[movie_indices].tolist()

    # Fetch the movie posters for recommendations
    recommended_movie_posters = []
    for i in movie_indices:
        movie_id = data_movies.iloc[i]['id']
        recommended_movie_posters.append(fetch_poster(movie_id))

    return recommended_movie_names, recommended_movie_posters


# Streamlit UI
st.title('Movie Recommender System')

# Movie selection dropdown
movie_list = data_movies['title'].values
selected_movie = st.selectbox(
    "Select a movie:",
    movie_list
)

if st.button('Get Recommendations'):
    recommendations, posters = get_recommendations(selected_movie)

    # Display recommended movies and posters
    for movie, poster in zip(recommendations, posters):
        st.image(poster, caption=movie, use_column_width=True)
