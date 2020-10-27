import pandas as pd
import numpy as np
import random

from correlations import pearson_correlations, similar_users

NUMBER_OF_USERS = 5
CORRELATION_THRESHOLD = 0.2
SIMILAR_USERS_MAX = 50

ratingsDF = pd.read_csv('movielens-small/ratings.csv')
ratingsDF.drop(['timestamp'], axis=1, inplace=True)

def calculate_recommendations(ratingsDF, correlationsDF, userId):
    # prediction function:
    # pred(a,p) = avg(r_a) + sum( sim(a,b)*(r_b,p - avg(r_b)) ) / sum(sim(a,b))

    # calculate average rating for each user, and rename column
    avg = ratingsDF.groupby('userId').mean().rename(columns={'rating':'average rating, user'})['average rating, user']
    print('AVERAGE RATINGS')
    print(avg)

    # merge correlation values to ratings 
    df = correlationsDF.merge(
        ratingsDF, left_on='userId', right_on='userId', how='inner')
    print('CORRELATIONS MERGED TO RATINGS')
    print(df)
   
    # merge average ratings to ratings
    df = df.merge(avg, left_on='userId', right_on='userId', how='inner')
    print('AVERAGE RATINGS MERGED TO RATINGS')
    print(df)

    # calculate adjusted ratings and add it as a column
    df['adjusted rating'] = df['PearsonCorr'] * (df['rating'] - df['average rating, user'])
    print('ADJUSTED RATINGS ADDED TO RATINGS')
    print(df)

    # Applies a sum to the topUsers after grouping it up by userId
    # group by movieId and calculate sum columns 'PearsonCorr', 'weighted rating'
    tempValuesDF = df.groupby('movieId').sum()[
        ['PearsonCorr', 'adjusted rating']]

    # rename columns
    tempValuesDF.columns = ['sum_PearsonCorr', 'sum_adjusted_rating']
    tempValuesDF['sum_adjusted_rating / sum_PearsonCorr'] = tempValuesDF['sum_adjusted_rating'] / tempValuesDF['sum_PearsonCorr']
    print('TEMP TABLE FOR CALCULATIONS')
    print(tempValuesDF)

    # create recommendation dataframe
    recommendationDF = pd.DataFrame()
    recommendationDF['recommendation score'] = avg[userId] + tempValuesDF['sum_adjusted_rating'] / tempValuesDF['sum_PearsonCorr']
    recommendationDF['movieId'] = tempValuesDF.index
    recommendationDf = recommendationDF.sort_values(by='recommendation score', ascending=False, inplace=True)
    print('RECOMMENDATIONS')
    print(recommendationDF)

    from sklearn.preprocessing import MinMaxScaler
    ratingScale = ratingsDF['rating'].unique()
    ratingScale.sort()
    # scale of ratings = tuple of (lowest rating, highest rating)
    ratingScale = (ratingScale[0], ratingScale[len(ratingScale) - 1])
    scaler = MinMaxScaler(feature_range=(ratingScale))
    recommendationDF['prediction for user ' + str(userId)] = scaler.fit_transform(recommendationDF['recommendation score'].values.reshape(-1,1))
    return recommendationDF

# group recommendation list with aggregation methods: average and least misery
def calculate_group_recommendation_list(recommendationLists):
    tempGroupRecommendationDF = recommendations[0]
    
    for i in range(1, len(recommendationLists)):
        tempGroupRecommendationDF = tempGroupRecommendationDF.merge(recommendations[i], left_on='movieId', right_on='movieId', how='outer', suffixes=(i - 1, i))

    columns = [col for col in tempGroupRecommendationDF.columns if 'prediction' in col or col == 'movieId']
    groupRecommendationDF = tempGroupRecommendationDF[columns]

    # remove rows with NaN values
    # in other words: we only consider the movies, that have a predicted score for each user in the group
    groupRecommendationDF = groupRecommendationDF.dropna()

    # calculate the average score, and add new column
    groupRecommendationDF.insert(1, 'average', groupRecommendationDF.iloc[:, 1:].mean(axis=1))
    groupListSorted = groupRecommendationDF.sort_values(by=['average'], ascending=False)

    # calculate the least misery score, and add new column
    groupRecommendationDF.insert(2, 'least misery', groupRecommendationDF.iloc[:, 2:].min(axis=1))
    groupListSorted = groupRecommendationDF.sort_values(by=['least misery'], ascending=False)

    return groupListSorted

# hybrid group recommendation list
def calculate_group_recommendation_list_hybrid(recommendationLists, alfa):
    tempGroupRecommendationDF = recommendations[0]
    
    for i in range(1, len(recommendationLists)):
        tempGroupRecommendationDF = tempGroupRecommendationDF.merge(recommendations[i], left_on='movieId', right_on='movieId', how='outer', suffixes=(i - 1, i))

    columns = [col for col in tempGroupRecommendationDF.columns if 'prediction' in col or col == 'movieId']
    groupRecommendationDF = tempGroupRecommendationDF[columns]

    # remove rows with NaN values
    # in other words: we only consider the movies, that have a predicted score for each user in the group
    groupRecommendationDF = groupRecommendationDF.dropna()

    # calculate the average score, and add new column
    groupRecommendationDF.insert(1, 'average', groupRecommendationDF.iloc[:, 1:].mean(axis=1))

    # calculate the least misery score, and add new column
    groupRecommendationDF.insert(2, 'least misery', groupRecommendationDF.iloc[:, 2:].min(axis=1))

    if (alfa == 0):
        groupRecommendationDF.insert(1, 'result', groupRecommendationDF['average'])

    groupListSorted = groupRecommendationDF.sort_values(by=['result'], ascending=False)
    return groupListSorted

def calculate_satisfaction(groupRecommendationList, users, k):
    #resultGroup = groupRecommendationList[['movieId','result']][:10]
    #resultGroup.index = np.arange(1, len(resultGroup) + 1)
    #resultUsers = []
    satisfaction = {}
    for i in range(0, len(users)):
        user = users[i]
        column = [col for col in groupRecommendationList.columns if col == 'prediction for user ' + str(user)]
        predictedScoreSumGroupList = groupRecommendationList[column][:k].sum()
        predictedScoreSumOwnList = groupRecommendationList[column].sort_values(by=column[0], ascending=False)[:k].sum()
        satisfaction[user] = predictedScoreSumGroupList.array[0] / predictedScoreSumOwnList.array[0]
    print('satisfaction scores:')
    print(satisfaction)
    print('for group recommendation list:')
    print(groupRecommendationList)

#####################################

allUsers = ratingsDF['userId'].unique().tolist()
users = []
# pick random users
for i in range(0, NUMBER_OF_USERS):
    user = random.choice(allUsers)
    users.append(user)
    allUsers.remove(user)

### ROUND 1
# at first round, the average aggregation is implemented (alfa = 0)

# calculate individual recommendation lists and add to a list
recommendations = []
for i in range(0,len(users)):
    correlationsDF = similar_users(ratingsDF, users[i], SIMILAR_USERS_MAX, CORRELATION_THRESHOLD)
    recommendations.append(calculate_recommendations(ratingsDF, correlationsDF, users[i]))

groupList = calculate_group_recommendation_list_hybrid(recommendations, 0)
print('\n RESULT: hybrid')
print(groupList)

# calculate satisfaction scores, use only top-k items in the group recommendation list
k = 10
satisfaction = calculate_satisfaction(groupList, users, k)
