import pandas as pd
import numpy as np
import random
import math

ratingsDF = pd.read_csv(
    'movielens-large/ratings.csv').drop(['timestamp'], axis=1)
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
    # finally, remove those users, that have negatice correlation value
    return correlationsDF[correlationsDF['PearsonCorr'] > 0]

# get similar users for target user (userId)
# function will return top 'n' users based on the number of movies watched in common


def similar_users(df, userId, n):
    print('\ntarget user: ', userId)
    targetUserRatings = df[df['userId'] == userId]

    print('\nsubset of users that have watched the same movies as the target user')
    condition1 = df['userId'] != userId
    condition2 = df['movieId'].isin(targetUserRatings['movieId'].tolist())
    userSubset = df[condition1 & condition2]

    # sort in descending order
    userSubsetSorted = sorted(userSubset.groupby(
        ['userId']), key=lambda x: len(x[1]), reverse=True)

    return pearson_correlations(targetUserRatings, userSubsetSorted[:n])


def calculate_ratings(ratingsDF, correlationsDF):
    df = correlationsDF.merge(
        ratingsDF, left_on='userId', right_on='userId', how='inner')
    # calculate weighted ratings and add it as a column
    df['weighted rating'] = df['PearsonCorr'] * df['rating']
    print(df)

    # Applies a sum to the topUsers after grouping it up by userId
    tempTopUsersRating = df.groupby('movieId').sum()[
        ['PearsonCorr', 'weighted rating']]
    tempTopUsersRating.columns = ['sum_PearsonCorr', 'sum_weighted_rating']

    recommendationDF = pd.DataFrame()
    recommendationDF['recommendation score'] = tempTopUsersRating['sum_weighted_rating']/tempTopUsersRating['sum_PearsonCorr']
    recommendationDF['movieId'] = tempTopUsersRating.index
    recommendationDf = recommendationDF.sort_values(by='recommendation score', ascending=False, inplace=True)
    print(recommendationDF)


correlationsDF = similar_users(ratingsDF, 1, 100)
calculate_ratings(ratingsDF, correlationsDF)
