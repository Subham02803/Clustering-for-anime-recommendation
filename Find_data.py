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

v0 = c0.sort_values(ascending=False).index
v1 = c1.sort_values(ascending=False).index
v2 = c2.sort_values(ascending=False).index
v3 = c3.sort_values(ascending=False).index


cluster = -1
def get_reated_anime(name):
    m = len(v0)
    for i,s in enumerate(v0):
        if (name == s):
            if(m > i):
                cluster = 0
                m = i
            break

    for i,s in enumerate(v1):
        if(name == s):
            if(m > i):
                cluster = 1
                m = i
            break

    for i, s in enumerate(v2):
        if (name == s):
            if (m > i):
                cluster = 2
                m = i
            break

    for i, s in enumerate(v3):
        if (name == s):
            if (m > i):
                cluster = 3
                m = i
            break

    if (cluster == 0):
        values = c0.sort_values(ascending=False)[0:5].index
        return values

    if (cluster == 1):
        values = c1.sort_values(ascending=False)[0:5].index
        return values

    if (cluster == 2):
        values = c2.sort_values(ascending=False)[0:5].index
        return values

    if(cluster == 3):
        values = c3.sort_values(ascending=False)[0:5].index
        return values
