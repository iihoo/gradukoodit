import pandas as pd
import numpy as np
import random
import math

ratingsDF = pd.read_csv('movielens-small/ratings.csv')
ratingsDF.drop(['timestamp'], axis=1, inplace=True)
moviesDF = pd.read_csv('movielens-small/movies.csv')

# calculate Pearson Correlation value between the target user and candidate users
# returns a dataframe with columns PearsonCorr and userId
def pearson_correlations(targetUserRatings, similarUserCandidates):
    pearsonCorrelationDict = {}

    for userId, ratings in similarUserCandidates:
        ratings = ratings.sort_values(by='movieId')
        targetUserRatings = targetUserRatings.sort_values(by='movieId')

        numberOfRatings = len(ratings)
        moviesInCommon = targetUserRatings[targetUserRatings['movieId'].isin(
            ratings['movieId'].tolist())]
        ratingsOfMoviesInCommonByTargetUser = moviesInCommon['rating'].tolist()
        ratingsOfOtherUser = ratings['rating'].tolist()

        Sxx = sum([i**2 for i in ratingsOfMoviesInCommonByTargetUser]) - \
            pow(sum(ratingsOfMoviesInCommonByTargetUser), 2) / \
            float(numberOfRatings)
        Syy = sum([i**2 for i in ratingsOfOtherUser]) - \
            pow(sum(ratingsOfOtherUser), 2)/float(numberOfRatings)
        Sxy = sum(i*j for i, j in zip(ratingsOfMoviesInCommonByTargetUser, ratingsOfOtherUser)) - \
            sum(ratingsOfMoviesInCommonByTargetUser) * \
            sum(ratingsOfOtherUser)/float(numberOfRatings)

        if Sxx != 0 and Syy != 0:
            pearsonCorrelationDict[userId] = Sxy/math.sqrt(Sxx*Syy)
        else:
            pearsonCorrelationDict[userId] = 0

    # create a correlation dataframe and sort according to the correlation value (descending)
    correlationsDF = pd.DataFrame.from_dict(
        pearsonCorrelationDict, orient='index')
    correlationsDF.columns = ['PearsonCorr']
    correlationsDF['userId'] = correlationsDF.index
    correlationsDF.index = range(len(correlationsDF))
    correlationsDF.sort_values(by='PearsonCorr', ascending=False, inplace=True)

    # NOTE check if necessary..
    # finally, remove those users, that have a correlation value lower than threshold
    correlationThreshold = 0.2
    return correlationsDF[correlationsDF['PearsonCorr'] > correlationThreshold]

# get similar users for target user (userId)
# function will return max 'n' users
def similar_users(df, userId, maxSimilarUsers):
    print('\ntarget user: ', userId)
    targetUserRatings = df[df['userId'] == userId]

    print('\nsubset of users that have watched the same movies as the target user')
    condition1 = df['userId'] != userId
    condition2 = df['movieId'].isin(targetUserRatings['movieId'].tolist())
    userSubset = df[condition1 & condition2]

    # sort in descending order
    userSubsetSorted = sorted(userSubset.groupby(
        ['userId']), key=lambda x: len(x[1]), reverse=True)

    return pearson_correlations(targetUserRatings, userSubsetSorted[:maxSimilarUsers])


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
    recommendationDF['recommendation score, scaled'] = scaler.fit_transform(recommendationDF['recommendation score'].values.reshape(-1,1))
    return recommendationDF

# aggregation method: average
def calculate_group_recommendation_list_average(recommendationLists):
    tempGroupRecommendationDF = recommendations[0]
    
    for i in range(1, len(recommendationLists)):
        tempGroupRecommendationDF = tempGroupRecommendationDF.merge(recommendations[i], left_on='movieId', right_on='movieId', how='outer', suffixes=(i - 1, i))

    columns = [col for col in tempGroupRecommendationDF.columns if 'recommendation score, scaled' in col or col == 'movieId']
    groupRecommendationDF = tempGroupRecommendationDF[columns]

    # remove rows with NaN values
    # in other words: we only consider the movies, that have a predicted score for each user in the group
    groupRecommendationDF = groupRecommendationDF.dropna()
    print(groupRecommendationDF)

    # calculate the average score, and add new column
    groupRecommendationDF.insert(1, 'average', groupRecommendationDF.iloc[:, 1:].mean(axis=1))
    groupListSorted = groupRecommendationDF.sort_values(by=['average'], ascending=False)
    return groupListSorted

# aggregation method: least misery
def calculate_group_recommendation_list_least_misery(recommendationLists):
    tempGroupRecommendationDF = recommendations[0]
    
    for i in range(1, len(recommendationLists)):
        tempGroupRecommendationDF = tempGroupRecommendationDF.merge(recommendations[i], left_on='movieId', right_on='movieId', how='outer', suffixes=(i - 1, i))

    columns = [col for col in tempGroupRecommendationDF.columns if 'recommendation score, scaled' in col or col == 'movieId']
    groupRecommendationDF = tempGroupRecommendationDF[columns]

    # remove rows with NaN values
    # in other words: we only consider the movies, that have a predicted score for each user in the group
    groupRecommendationDF = groupRecommendationDF.dropna()

    # calculate the least misery score, and add new column
    groupRecommendationDF.insert(1, 'least misery', groupRecommendationDF.iloc[:, 1:].min(axis=1))
    groupListSorted = groupRecommendationDF.sort_values(by=['least misery'], ascending=False)
    return groupListSorted

#####################################

allUsers = ratingsDF['userId'].unique().tolist()
users = []
# pick random users
for i in range(0,4):
    user = random.choice(allUsers)
    users.append(user)
    allUsers.remove(user)

# add individual recommendation lists to a list
recommendations = []
for i in range(0,len(users)):
    correlationsDF = similar_users(ratingsDF, users[i], 50)
    recommendations.append(calculate_recommendations(ratingsDF, correlationsDF, users[i]))

# aggregate the group recommendation list
groupList = calculate_group_recommendation_list_average(recommendations)
print('\n RESULT: average aggregation')
print(groupList)

groupListLeastMisery = calculate_group_recommendation_list_least_misery(recommendations)
print('\n RESULT: least misery aggregation')
print(groupListLeastMisery)

