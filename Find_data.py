import pandas as pd
import pickle

anime = pd.read_csv('anime.csv')
anime_name = anime['name']
anime_genre = anime['genre']

pkl_file = open('myfile.pkl', 'rb')
df = pickle.load(pkl_file)
pkl_file.close()

def get_genre(name):
    for i,x in enumerate(anime_name.values):
        if(x == name):
            return anime_genre[i], i

    str = 'Error'
    return str, -1

def get_reated_anime(i, name):
