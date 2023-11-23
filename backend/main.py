from flask import Flask, request, jsonify, render_template
from tmdbv3api import TMDb
from tmdbv3api import Movie
import requests
import TMDB as t
import math

app = Flask(__name__)

def fetch_movie_results(movie_name):
    tmdb = TMDb()
    movie = Movie()
    tmdb.api_key = 'd3411de23358daa43bbe0657545a26de'
    tmdb.language = 'en'
    tmdb.debug = True

    result_list = []
    print(f"Movie Name: {movie_name}")
    recommedations = t.improved_recommendations(movie_name)

    for name in recommedations:
        print(f"\n\nFetching the API search for movie {name}: \n\n")
        search = movie.search(name)
        search = [x for x in search if hasattr(x, 'vote_average') and x.vote_average is not None]
        if search:
            search = search[0]
            dict_Temp = {
                'id' : search['id'],
                'poster_path' : search['poster_path'],
                'title' : search['title'],
                'description' : search['overview'],
                'vote_average' : round(search['vote_average'], 1)
            }
            result_list.append(dict_Temp)

    return result_list

# @app.route("/") 
# def hello(): 
#     return render_template('index.html') 

@app.route('/recommend', methods=['POST'])
def recommend_movies():
    data = request.get_json()
    print(data)
    user_query = data['value']
    print(user_query)

    user_movie_data = fetch_movie_results(user_query)
    return jsonify(user_movie_data)

if __name__ == "__main__":
    app.run(debug=True)