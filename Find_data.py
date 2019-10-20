import pandas as pd
import pickle

anime = pd.read_csv('anime.csv')
anime_name = anime['name']
anime_genre = anime['genre']

pkl_file0 = open('C0File.pkl', 'rb')
c0 = pickle.load(pkl_file0)
pkl_file0.close()

pkl_file1 = open('C1File.pkl', 'rb')
c1 = pickle.load(pkl_file1)
pkl_file1.close()

pkl_file2 = open('C2File.pkl', 'rb')
c2 = pickle.load(pkl_file2)
pkl_file2.close()

pkl_file3 = open('C3File.pkl', 'rb')
c3 = pickle.load(pkl_file3)
pkl_file3.close()

def get_genre(name):
    for i,x in enumerate(anime_name.values):
        if(x == name):
            return anime_genre[i], i

    str = 'Error'
    return str, -1

cluster = -1
def get_reated_anime(name):
    for s in c0.index:
        if(name == s):
            cluster = 0
            break

    if(cluster == 0):
        values = c0.sort_values(ascending=False)[0:5].index
        return values

    for s in c1.index:
        if(name == s):
            cluster = 1
            break

    if(cluster == 1):
        values = c1.sort_values(ascending=False)[0:5].index
        return values

    for s in c2.index:
        if(name == s):
            cluster = 2
            break

    if(cluster == 2):
        values = c2.sort_values(ascending=False)[0:5].index
        return values

    for s in c3.index:
        if(name == s):
            cluster = 3
            break

    if(cluster == 3):
        values = c3.sort_values(ascending=False)[0:5].index
        return values