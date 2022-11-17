from flask import Flask, jsonify, request
import pickle
import pandas as pd
import numpy as np
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
count = CountVectorizer()

app = Flask(__name__)

movie_dict = pickle.load(
    open('movie_name.pkl', 'rb'))
movie_list = pd.DataFrame(movie_dict)

movie_detail_dict = pickle.load(
    open('movie_detail.pkl', 'rb'))
movie_details = pd.DataFrame(movie_detail_dict)
final_rec = pd.DataFrame()


def fetch_poster(movie_id):
    response = requests.get(
        'https://api.themoviedb.org/3/movie/{}?api_key=6d5195f330d79be412d6ad8ee5cc8f20'.format(movie_id))
    data = response.json()
    poster_path = data['poster_path']
    return "https://image.tmdb.org/t/p/original/" + poster_path


def get_title_from_index(index):
    return movie_list[movie_list.index == index]["title"].values[0]


def get_index_from_title(movie):
    return movie_list[movie_list["title"] == movie].index[0]


@app.route('/')
def home():
    return "hello world"


@app.route('/check', methods=['POST'])
def checking():
    # data = request.json
    movie_name = request.form.get('movie_name')
    rec_type2 = request.form.get('rec_type')
    rec_type = rec_type2.split()
    return jsonify(movie_name, rec_type2)


@app.route('/predict', methods=['POST'])
def predict():
    movie_name = request.form.get('movie_name')
    rec_type2 = request.form.get('rec_type')
    rec_type = rec_type2.split()
    if len(rec_type) == 0:
        rec_type.append('Overall')
    global final_rec
    movie_idx = get_index_from_title(movie_name)
    final_rec = movie_details[rec_type[0]]
    for t in rec_type[1:]:
        final_rec += movie_details[t]
    matrix = count.fit_transform(final_rec)
    cosine_sim = cosine_similarity(matrix, matrix)

    similar_movies = list(enumerate(cosine_sim[movie_idx]))
    sorted_similar_movie = sorted(
        similar_movies, key=lambda x: x[1], reverse=True)
    recommended_movie = []
    recommended_movie_poster = []
    for i in sorted_similar_movie[1:16]:
        recommended_movie.append(get_title_from_index(i[0]))
        index = get_index_from_title(get_title_from_index(i[0]))
        recommended_movie_poster.append(
            fetch_poster(movie_list.iloc[index]['id']))

    result = {'movie_name': recommended_movie,
              'movie_poster': recommended_movie_poster}
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)
