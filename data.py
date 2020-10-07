import pandas as pd
import random

ratings = pd.read_csv('movielens-large/ratings.csv')
movies = pd.read_csv('movielens-large/movies.csv')

# remove timestamp from the ratings data
ratings.drop('timestamp', axis=1, inplace=True)
#print(ratings)

# number of unique users in the data
#ratings['userId'].nunique()

# number of unique movies in the data
#movies['movieId'].nunique()

# create the group by choosing random members
# note: avoid duplicate members
#group = [random.randint(1,ratings['userId'].nunique()), random.randint(1,ratings['userId'].nunique()), random.randint(1,ratings['userId'].nunique())]
#print(group)

# df index = userId, columns = movieId
# NOTE should NAN values be filled with 0?
# NOTE we could insert values to the matrix as user-movie 'rows' OR as movie-user 'columns'
# --> check which way is more efficient
#df = pd.DataFrame(index = ratings['userId'].unique(), columns = ratings['movieId'].unique())
df = pd.DataFrame(index = ratings['userId'].unique())
df.index.name = 'userId'

#df = pd.DataFrame(columns = ratings['movieId'].unique())

#ratings for user 1
##condition = ratings['userId'] == 1
# take columns rating and movieId, where userId = 1, the set movieId as index
##r_1 = ratings[condition][['movieId', 'rating']].set_index('movieId')
##print(r_1)

##result = df.append(pd.Series(r_1['rating'], index=r_1.index), ignore_index=True)
##result.index += 1
##print(result)

### try with adding columns movie-user to the matrix

# ratings for movie 1
condition = ratings['movieId'] == 1
# take columns movieId and userId, where movieId = 1, the set userId as index
m_1 = ratings[condition][['userId', 'rating']].set_index('userId').rename(columns={'rating' : 1})
#m_1.index.name = None

print(ratings)

print('movie 1 ratings')
print(m_1)

# update original matrix
#df.update(m_1)
df.insert(loc=len(df.columns), column = 1, value=m_1)
print(df)
