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