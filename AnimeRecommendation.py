#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

get_ipython().run_line_magic('matplotlib', 'inline')


plt.rcParams['figure.figsize'] = (6, 4)
plt.style.use('ggplot')
get_ipython().run_line_magic('config', "InlineBackend.figure_formats = {'png', 'retina'}")


# In[2]:


anime = pd.read_csv('anime.csv')
anime.head()


# In[3]:


user = pd.read_csv('rating.csv')
user.sample(10)


# In[4]:


print("Shape of anime dataset {}".format(anime.shape))


# In[5]:


print("Shape of user dataset {}".format(user.shape))


# In[6]:


user[user['user_id'] == 1].rating.mean()


# In[7]:


user[user['user_id'] == 7].rating.mean()


# In[8]:


Mean_Rate = user.groupby(['user_id']).mean().reset_index()
print(Mean_Rate.head(10))
print("Shape of this datset is {}".format(Mean_Rate.shape))


# In[9]:


Mean_Rate['mean_rating'] = Mean_Rate['rating']
print(Mean_Rate.head())


# In[10]:


Mean_Rate.drop(['anime_id','rating'], axis=1,inplace=True)
print(Mean_Rate.head())


# In[11]:


user = pd.merge(user, Mean_Rate, on=['user_id', 'user_id'])
print(user.head())


# In[12]:


#If user rating is less then mean_rating we drop that column as we are looking which anime is liked by the user
user = user.drop(user[user.rating < user.mean_rating].index)


# In[13]:


print(user[user['user_id'] == 1])


# In[14]:


print(user[user['user_id'] == 7])


# In[15]:


print("Updated shape of the user is now {}".format(user.shape))


# In[16]:


unique_user = np.unique(user['user_id'])
print(unique_user)
print(len(unique_user))


# In[17]:


user = user.rename({'rating':'userRating'}, axis='columns')
print(user.head())


# In[54]:


mergedata = pd.merge(anime, user, on=['anime_id','anime_id'])
mergedata= mergedata[mergedata.user_id <= 20000]
mergedata.head()


# In[19]:


unique_user = np.unique(mergedata['user_id'])
print(unique_user)
print(len(unique_user))


# In[20]:


unique_anime = np.unique(mergedata['anime_id'])
print(unique_anime)
print(len(unique_anime))
unique_anime = np.unique(mergedata['name'])
print(unique_anime)
print(len(unique_anime))


# In[22]:


#Now we are going to create a crosstable of 'user_id' and 'name' using pandas.crosstab
#here we use 'user_id' as row or index and 'name' as column
user_anime = pd.crosstab(mergedata['user_id'], mergedata['name'])
user_anime.sample(5)


# In[23]:


user_anime.shape


# In[24]:


from sklearn.decomposition import PCA
pca = PCA(n_components=3)
pca.fit(user_anime)
pca = pca.transform(user_anime)


# In[56]:


ps = pd.DataFrame(pca)
ps.head()


# In[57]:


ps.shape


# In[58]:


#Cluster
tocluster = pd.DataFrame(ps[[0,1,2]])
tocluster.head()


# In[59]:


plt.rcParams['figure.figsize'] = (16, 9)

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(tocluster[0], tocluster[1], tocluster[2])

plt.title('Data points in 3D PCA axis', fontsize=20)
plt.show()


# In[31]:


from sklearn.cluster import KMeans

no_of_cluster = range(2,10)
kmeans = [KMeans(n_clusters=i) for i in no_of_cluster]
print(kmeans)


# In[32]:


#Score of every no of cluster
score = [kmeans[i].fit(tocluster).score(tocluster) for i in range(len(kmeans))]
print(score)


# In[33]:


plt.plot(no_of_cluster, score)
plt.xlabel('no_of_clusters')
plt.ylabel('score')
plt.show()


# In[37]:


#Silhouette score
from sklearn.metrics import silhouette_score
score = []
inertia_list = np.empty(10)

for i in range(2,10):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(tocluster)
    inertia_list[i] = kmeans.inertia_
    score.append(silhouette_score(tocluster, kmeans.labels_))


# In[39]:


plt.plot(range(2,10), score);
plt.title('Results KMeans')
plt.xlabel('n_clusters');
plt.axvline(x=4, color='blue', linestyle='--')
plt.ylabel('Silhouette Score');
plt.show()


# In[46]:


from sklearn.cluster import KMeans
cluster = KMeans(n_clusters = 4, random_state=35).fit(tocluster)
centers = cluster.cluster_centers_
c_preds = cluster.predict(tocluster) #predicterd point

print(centers)


# In[48]:


fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(tocluster[0], tocluster[1], tocluster[2], c=c_preds)
for center in centers:
    ax.scatter(center[0], center[1], center[2], color='red')

plt.title('Data points in 3D PCA axis with centers', fontsize=20)
plt.show()


# In[51]:


fig = plt.figure(figsize=(10,8))
plt.scatter(tocluster[1],tocluster[0], c=c_preds)

for center in centers:
    plt.scatter(center[1], center[0], color='red')
    
plt.title('Data points in 2D PCA axis with centers', fontsize=20)
plt.show()


# In[52]:


user_anime['clusters'] = c_preds
user_anime.head()


# In[53]:


user_anime.info()


# In[60]:


#unique c_preds
unique_c_preds = np.unique(c_preds)
print(unique_c_preds)


# In[62]:


#Characteristic of each cluster
c0 = user_anime[user_anime['clusters'] == 0].drop('clusters', axis=1).mean()
c1 = user_anime[user_anime['clusters'] == 1].drop('clusters', axis=1).mean()
c2 = user_anime[user_anime['clusters'] == 2].drop('clusters', axis=1).mean()
c3 = user_anime[user_anime['clusters'] == 3].drop('clusters', axis=1).mean()


# In[64]:


#Top 15 anime which will explain characteristic of this cluster
c0.sort_values(ascending=False)[0:15]


# In[68]:


#Create anime info list fucntion

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


# In[69]:


animelist = list(c0.index)
data = pd.DataFrame()
data['genre'],data['episode'],data['rating'],data['member'] =  Anime_Info_List(animelist)
data['genre']


# In[72]:


#Find different types of gerne

set_keywords = set()
for x in data['genre'].str.split(',').values:
    if isinstance(x, float):
        continue
    #The isinstance() function returns True if the specified object is of the specified type, otherwise False.
    set_keywords = set_keywords.union(x)
    
set_keywords


# In[74]:


len(set_keywords)


# In[75]:


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
        


# In[82]:


from wordcloud import WordCloud

def makeCloud(Dict,name,color):
    words = dict()

    for s in Dict:
        words[s[0]] = s[1]

        wordcloud = WordCloud(
                      width=1500,
                      height=500, 
                      background_color=color, 
                      max_words=20,
                      max_font_size=500, 
                      normalize_plurals=False)
        wordcloud.generate_from_frequencies(words)


    fig = plt.figure(figsize=(12, 8))
    plt.title(name)
    plt.imshow(wordcloud)
    plt.axis('off')

    plt.show()


# In[87]:


#####........FOR C0....#######

c0_animeList = list(c0.sort_values(ascending=False)[0:15].index)
c0_data = pd.DataFrame()
c0_data['genre'],c0_data['episode'],c0_data['rating'],c0_data['member'] =  Anime_Info_List(c0_animeList)
c0_data.iloc[:,1:4] = c0_data.iloc[:,1:4].astype(int) # change to numeric object to integer
keyword_occurences, dum = count_word(c0_data, 'genre', set_keywords)
makeCloud(keyword_occurences[0:10],"cluster 0","#ccffff")


# In[88]:


keyword_occurences[0:10]


# In[89]:


print('cluster 0\nAVG episode : {0}\nAVG movie rating : {1}\nAVG member : {2}'.format(c0_data['episode'].mean(), c0_data['rating'].mean(),c0_data['member'].mean()))


# In[90]:


#####........FOR C1....#######
c1.sort_values(ascending=False)[0:15]


# In[93]:


c1_animeList = list(c1.sort_values(ascending=False)[0:15].index)
c1_data = pd.DataFrame()
c1_data['genre'],c1_data['episode'],c1_data['rating'],c1_data['member'] =  Anime_Info_List(c1_animeList)
c1_data.iloc[:,1:4] = c1_data.iloc[:,1:4].astype(int) # change to numeric object to integer
keyword_occurences, dum = count_word(c1_data, 'genre', set_keywords)
makeCloud(keyword_occurences[0:10],"cluster 1","#ffccff")


# In[94]:


keyword_occurences[0:10]


# In[95]:


print('cluster 1\nAVG episode : {0}\nAVG movie rating : {1}\nAVG member : {2}'.format(c1_data['episode'].mean(), c1_data['rating'].mean(),c1_data['member'].mean()))


# In[97]:


#####........FOR C2....#######
c2.sort_values(ascending=False)[0:15]


# In[99]:


c2_animeList = list(c2.sort_values(ascending=False)[0:15].index)
c2_data = pd.DataFrame()
c2_data['genre'],c2_data['episode'],c2_data['rating'],c2_data['member'] =  Anime_Info_List(c2_animeList)
c2_data.iloc[:,1:4] = c2_data.iloc[:,1:4].astype(int) # change to numeric object to integer
keyword_occurences, dum = count_word(c2_data, 'genre', set_keywords)
makeCloud(keyword_occurences[0:10],"cluster 2","#ffffcc")


# In[100]:


keyword_occurences[0:10]


# In[101]:


print('cluster 2\nAVG episode : {0}\nAVG movie rating : {1}\nAVG member : {2}'.format(c2_data['episode'].mean(), c2_data['rating'].mean(),c2_data['member'].mean()))


# In[102]:


#####........FOR C3....#######
c3.sort_values(ascending=False)[0:15]


# In[103]:


c3_animeList = list(c3.sort_values(ascending=False)[0:15].index)
c3_data = pd.DataFrame()
c3_data['genre'],c3_data['episode'],c3_data['rating'],c3_data['member'] =  Anime_Info_List(c3_animeList)
c3_data.iloc[:,1:4] = c3_data.iloc[:,1:4].astype(int) # change to numeric object to integer
keyword_occurences, dum = count_word(c3_data, 'genre', set_keywords)
makeCloud(keyword_occurences[0:10],"cluster 3","#fff2e6")


# In[104]:


keyword_occurences[0:10]


# In[105]:


print('cluster 3\nAVG episode : {0}\nAVG movie rating : {1}\nAVG member : {2}'.format(c3_data['episode'].mean(), c3_data['rating'].mean(),c3_data['member'].mean()))

