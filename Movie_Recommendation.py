import pandas as pd
import numpy as np
from IPython import get_ipython
import warnings
warnings.filterwarnings("ignore")

movies = pd.read_csv(r'C:/Users/USER/Desktop/Project recommendation/tmdb_5000_movies.csv')
credits = pd.read_csv(r'C:/Users/USER/Desktop/Project recommendation/tmdb_5000_credits.csv')

movies.head()
movies.tail()
movies.shape
movies.duplicated().sum()
movies.isnull().sum()
movies.info()
movies.describe()

credits.head()
credits.tail()
credits.shape
credits.duplicated().sum()
credits = credits[['movie_id', 'title', 'cast', 'crew']]
credits.info()
credits.describe()

movies.nunique()
credits.nunique()

#### Merging the data from second to first ########
movies = movies.merge(credits, on = 'title')
movies.head()
movies.shape

movies_df = movies[['id', 'title', 'overview', 'genres', 'cast', 'crew', 'keywords']]
movies_df.dropna(inplace = True)
movies_df.iloc[0].genres

import ast

def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L 

movies_df['genres'] = movies_df['genres'].apply(convert)
movies_df['genres']
movies_df.head()
movies_df['keywords'] = movies_df['keywords'].apply(convert)
movies_df['keywords']

movies_df.head()

movies_df.iloc[0].cast

def convert3(obj):
    L = []
    count = 0
    for i in ast.literal_eval(obj):
        if count < 3:
            L.append(i['name'])
        count +=1  
    return L

movies_df['cast'] = movies_df['cast'].apply(convert3)
movies_df['cast']
movies_df.head()
movies_df.iloc[0].overview
movies_df['overview'] = movies_df['overview'].apply(lambda x : x.split())
movies_df['overview']
movies_df.head()
movies_df.drop('crew', axis = 1)

def collapse(L):
    L1 = []
    for i in L:
        L1.append(i.replace(" ",""))
    return L1

movies_df['cast'] = movies_df['cast'].apply(collapse)
movies_df['genres'] = movies_df['genres'].apply(collapse)
movies_df['keywords'] = movies_df['keywords'].apply(collapse)

movies_df['tags'] = movies_df['overview']+movies_df['genres']+movies_df['keywords']+movies_df['cast']
movies_df_new = movies_df.copy()
movies_df_new = movies_df_new.drop('crew', axis = 1)

movies_df_new.head()
new_df = movies_df_new[['id','title','tags']]
new_df.head()
new_df['tags'] = new_df['tags'].apply(lambda x :" ".join(x))
new_df['tags']
new_df.head()
new_df['tags'][0]

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000 ,stop_words='english')

vectors = cv.fit_transform(new_df['tags']).toarray()

vectors.shape

cv.get_feature_names()
len(cv.get_feature_names())

from sklearn.metrics.pairwise import cosine_similarity
cosine_similarity(vectors).shape

similarity = cosine_similarity(vectors)
similarity[0]
sorted(list(enumerate(similarity[0])),reverse =True , key = lambda x:x[1])[1:6]

def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0] ##fetching the movie index
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)),reverse =True , key = lambda x:x[1])[1:6]
    
    for i in movie_list:
        print(new_df.iloc[i[0]].title)
        
recommend('The Dark Knight Rises')
recommend('Iron Man')
recommend('Avengers: Age of Ultron')

import pickle 
pickle.dump(new_df, open('movies_df.pkl','wb'))
pickle.dump(similarity, open('similarity.pkl','wb'))

new_df['title'].values
new_df.to_dict()

pickle.dump(new_df.to_dict(), open('movies_dict.pkl','wb'))


