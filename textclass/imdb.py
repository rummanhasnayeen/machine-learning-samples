import pandas as pd
import numpy as np

names_str=['budget', 'genres', 'homepage', 'id', 'keywords', 'original_language', 'original_title', 'overview', 'popularity', 'production_companies', 'production_countries', 'release_date', 'revenue', 'runtime', 'spoken_languages', 'status', 'tagline', 'title', 'vote_average', 'vote_count']
filepath = '/home/rumman/Downloads/data sets/tmdb-5000-movie-dataset/tmdb_5000_movies.csv'
df = pd.read_csv(filepath, header=None, names=names_str, sep=',')

df = df.drop('homepage', 1)
df = df.drop('id', 1)
df = df.drop('original_language', 1)
df = df.drop('original_title', 1)
df = df.drop('production_companies', 1)
df = df.drop('production_countries', 1)
df = df.drop('release_date', 1)
df = df.drop('runtime', 1)
df = df.drop('spoken_languages', 1)
# df = df.drop('original_title', 1)

a = (df['genres'].tolist())
b=[]
for text in a:
    print([name in text])


# print(df.genres[1])