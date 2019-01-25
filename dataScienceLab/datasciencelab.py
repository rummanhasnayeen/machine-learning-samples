import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# # Read the data into a dataframe
# data = pd.read_csv("/home/rumman/Desktop/DS_lab2/goodreads.csv")
#
# # Examine the first couple of rows of the dataframe
# bookinfo = data.head()
# print(bookinfo)
data = pd.read_csv("/home/rumman/Desktop/DS_lab2/goodreads.csv", header=None,
               names=["rating", 'review_count', 'isbn', 'booktype','author_url', 'year', 'genre_urls', 'dir','rating_count', 'name'],
)
#Examine the first couple of rows of the dataframe
# bookinfo = data.head()
# print(data.dtypes)

# print(data.shape)
# print(data.columns)

# number = np.sum([data.rating.isnull()])
# print(number)

data = data[data.year.notnull()]
#
# number = np.sum([data.year.isnull()])
# print(data)
# print(number)
# print(data.dtypes)
#

# print(np.sum(data.year.isnull()))
# print(np.sum(data.rating_count.isnull()))
# print(np.sum(data.review_count.isnull()))
# # How many rows are removed?
# print(data.shape)


#lets try to change the data types of rating count and year to integer
data.rating_count=data.rating_count.astype(int)
data.review_count=data.review_count.astype(int)
data.year=data.year.astype(int)
#
# print(data.dtypes)


#
# # Some of the other colums that should be strings have NaN.
data.loc[data.genre_urls.isnull(), 'genre_urls']=""

data.loc[data.isbn.isnull(), 'isbn']=""
#
# print(data.shape)
#Get the first author_url
test_string = data.author_url[0]
print(test_string)

# Isolate the author name
test_string.split('/')[-1].split('.')[1:][0]
# print(test_string[-2])

# Write a function that accepts an author url and returns the author's name based on your experimentation above
def get_author(url):
    name = url.split('/')[-1].split('.')[1:][0]
    return name


# Apply the get_author function to the 'author_url' column using '.map'
# and add a new column 'author' to store the names
data['author'] = data.author_url.map(get_author)
# print(data.author[0:5])


#Write a function that accepts a genre url and returns the genre name based on your experimentation above
def split_and_join_genres(url):
    genres = url.strip().split('|')
    genres = [e.split('/')[-1] for e in genres]
    return "|".join(genres)

data['genres'] = data.genre_urls.map(split_and_join_genres)
data.head()

testrow = data[data.author == "Marguerite_Yourcenar"]
# print(testrow)



# Generate histograms using the format data.COLUMN_NAME.hist(bins=YOUR_CHOICE_OF_BIN_SIZE)
# If your histograms appear strange or counter-intuitive, make appropriate adjustments in the data and re-visualize.

data.review_count.hist(bins=100)
plt.xlabel('Number of reviews')
plt.ylabel('Frequency')
plt.title('Number of reviews');

# plt.show();



plt.hist(data.year, bins=100);
plt.xlabel('Year written')
plt.ylabel('log(Frequency)')
plt.title('Number of books in a year')
# plt.show();


#It appears that some books were written in negative years!
# Print out the observations that correspond to negative years.
neg = data[data.year < 0]
# What do you notice about these books?
# print(neg)

for year, subset in data.groupby('year'):
    #Find the best book of the year
    bestbook = subset[subset.rating == subset.rating.max()]
    # if bestbook.shape[0] > 1:
    #     # print(year, bestbook.name.values, bestbook.rating.values)
    # else:
    #     # print(year, bestbook.name.values[0], bestbook.rating.values[0])
    #

#Get the unique genres contained in the dataframe.
genres = set()
for genre_string in data.genres:
    genres.update(genre_string.split('|'))
genres = sorted(genres)
# print(genres)

for genre in genres:
    data["genre:" + genre] = [genre in g.split('|') for g in data.genres]

genrehead = data.head()
# print(genrehead)
# print(data.shape)

genreslist = ['genre:'+g for g in genres]
dfg = data[genreslist].sum() # True's sum as 1's, and default sum is columnwise



# print(dfg)
# dfg.sort_values(ascending=False)
#
# print(dfg)
# dfg.sort_values(ascending=False).plot(kind = "bar");
#
# print(dfg)
# # The above histogram looks very clumsy!
# # so now view less data
# dfg.sort_values(ascending=False).iloc[0:20].plot(kind = "bar");
#
# print(dfg)
# dfg.sort_values(ascending=False)[0:10]
#
# print(dfg)
genres_wanted=dfg.index[dfg.values > 550]
print(genres_wanted.shape)
print(genres_wanted)
fig, axes = plt.subplots(nrows=10, ncols=3, figsize=(12, 40), tight_layout=True)
bins = np.arange(1950, 2013, 3)
for ax, genre in zip(axes.ravel(), genres_wanted):
    ax.hist(data[data[genre] == True].year.values, bins=bins, histtype='stepfilled', normed=True, color='r', alpha=.2,
            ec='none')
    ax.hist(data.year, bins=bins, histtype='stepfilled', ec='None', normed=True, zorder=0, color='#cccccc')

    ax.annotate(genre.split(':')[-1], xy=(1955, 3e-2), fontsize=14)
    ax.xaxis.set_ticks(np.arange(1950, 2013, 30))