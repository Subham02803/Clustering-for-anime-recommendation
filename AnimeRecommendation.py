import numpy as np
import pandas as pd 
import pickle
anime = pd.read_csv('anime.csv')
user = pd.read_csv('rating.csv')
Mean_Rate = user.groupby(['user_id']).mean().reset_index()
Mean_Rate['mean_rating'] = Mean_Rate['rating']
Mean_Rate.drop(['anime_id','rating'], axis=1,inplace=True)
user = pd.merge(user, Mean_Rate, on=['user_id', 'user_id'])
user = user.drop(user[user.rating < user.mean_rating].index)
unique_user = np.unique(user['user_id'])
user = user.rename({'rating':'userRating'}, axis='columns')
mergedata = pd.merge(anime, user, on=['anime_id','anime_id'])
mergedata= mergedata[mergedata.user_id <= 20000]
user_anime = pd.crosstab(mergedata['user_id'], mergedata['name'])
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
pca.fit(user_anime)
pca = pca.transform(user_anime)
ps = pd.DataFrame(pca)
tocluster = pd.DataFrame(ps[[0,1,2]])
from sklearn.cluster import KMeans
cluster = KMeans(n_clusters = 4, random_state=35).fit(tocluster)
centers = cluster.cluster_centers_
c_preds = cluster.predict(tocluster) #predicterd point
user_anime['clusters'] = c_preds
unique_c_preds = np.unique(c_preds)

#Characteristic of each cluster
c0 = user_anime[user_anime['clusters'] == 0].drop('clusters', axis=1).mean()
c1 = user_anime[user_anime['clusters'] == 1].drop('clusters', axis=1).mean()
c2 = user_anime[user_anime['clusters'] == 2].drop('clusters', axis=1).mean()
c3 = user_anime[user_anime['clusters'] == 3].drop('clusters', axis=1).mean()

output0 = open('C0File.pkl', 'wb')
pickle.dump(c0, output0)
output0.close()

output1 = open('C1File.pkl', 'wb')
pickle.dump(c1, output1)
output1.close()

output2 = open('C2File.pkl', 'wb')
pickle.dump(c2, output2)
output2.close()

output3 = open('C3File.pkl', 'wb')
pickle.dump(c3, output3)
output3.close()



'''

def Anime_Info_List(animelist):
    episode_list = list()
    genre_list = list()
    member_list = list()
    rating_list = list()
    
    for x in anime['name']:
        if x in animelist:
            episode_list.append(anime[anime['name'] == x].episodes.values.astype(int))
            member_list.append(anime[anime['name'] == x].members.values.astype(int))
            rating_list.append(anime[anime['name'] == x].rating.values.astype(float))
            
            for y in anime[anime['name'] == x].genre.values:
                genre_list.append(y)
                
    return genre_list, episode_list, rating_list, member_list

animelist = list(c0.index)
data = pd.DataFrame()
data['genre'],data['episode'],data['rating'],data['member'] =  Anime_Info_List(animelist)
set_keywords = set()
for x in data['genre'].str.split(',').values:
    if isinstance(x, float):
        continue
    #The isinstance() function returns True if the specified object is of the specified type, otherwise False.
    set_keywords = set_keywords.union(x)

def count_word(dataframe, col, keyword):
    keyword_count = dict()
    for s in keyword:
        keyword_count[s] = 0
    for x in dataframe[col].str.split(','):
        if type(x) == float and pd.isnull(x):
            continue
        for s in [s for s in x if s in keyword]:
            if pd.notnull(s):
                keyword_count[s] = keyword_count[s] + 1
    
    keyword_occurences = []
    for k,v in keyword_count.items():
        keyword_occurences.append([k,v])
        
    keyword_occurences.sort(key = lambda x:x[1], reverse = True)
    return keyword_occurences, keyword_count
c0_animeList = list(c0.sort_values(ascending=False)[0:15].index)
c0_data = pd.DataFrame()
c0_data['genre'],c0_data['episode'],c0_data['rating'],c0_data['member'] =  Anime_Info_List(c0_animeList)
c0_data.iloc[:,1:4] = c0_data.iloc[:,1:4].astype(int) # change to numeric object to integer
keyword_occurences, dum = count_word(c0_data, 'genre', set_keywords)
print('cluster 0\nAVG episode : {0}\nAVG movie rating : {1}\nAVG member : {2}'.format(c0_data['episode'].mean(), c0_data['rating'].mean(),c0_data['member'].mean()))
c1_animeList = list(c1.sort_values(ascending=False)[0:15].index)
c1_data = pd.DataFrame()
c1_data['genre'],c1_data['episode'],c1_data['rating'],c1_data['member'] =  Anime_Info_List(c1_animeList)
c1_data.iloc[:,1:4] = c1_data.iloc[:,1:4].astype(int) # change to numeric object to integer
keyword_occurences, dum = count_word(c1_data, 'genre', set_keywords)
print('cluster 1\nAVG episode : {0}\nAVG movie rating : {1}\nAVG member : {2}'.format(c1_data['episode'].mean(), c1_data['rating'].mean(),c1_data['member'].mean()))
c2_animeList = list(c2.sort_values(ascending=False)[0:15].index)
c2_data = pd.DataFrame()
c2_data['genre'],c2_data['episode'],c2_data['rating'],c2_data['member'] =  Anime_Info_List(c2_animeList)
c2_data.iloc[:,1:4] = c2_data.iloc[:,1:4].astype(int) # change to numeric object to integer
keyword_occurences, dum = count_word(c2_data, 'genre', set_keywords)
print('cluster 2\nAVG episode : {0}\nAVG movie rating : {1}\nAVG member : {2}'.format(c2_data['episode'].mean(), c2_data['rating'].mean(),c2_data['member'].mean()))
c3_animeList = list(c3.sort_values(ascending=False)[0:15].index)
c3_data = pd.DataFrame()
c3_data['genre'],c3_data['episode'],c3_data['rating'],c3_data['member'] =  Anime_Info_List(c3_animeList)
c3_data.iloc[:,1:4] = c3_data.iloc[:,1:4].astype(int) # change to numeric object to integer
keyword_occurences, dum = count_word(c3_data, 'genre', set_keywords)
print('cluster 3\nAVG episode : {0}\nAVG movie rating : {1}\nAVG member : {2}'.format(c3_data['episode'].mean(), c3_data['rating'].mean(),c3_data['member'].mean()))
'''