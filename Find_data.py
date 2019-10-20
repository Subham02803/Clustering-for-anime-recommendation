import pandas as pd

anime = pd.read_csv('anime.csv')
anime_name = anime['name']
anime_genre = anime['genre']

def get_genre(name):
    for i,x in enumerate(anime_name.values):
        if(x == name):
            return anime_genre[i], i

    str = 'Error'
    return str, -1

def get_reated_anime(i, name):
