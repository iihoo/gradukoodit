import math
import pandas as pd
import numpy as np

def similar_users(ratings, userId, moviesInCommonMinimum, correlationThreshold):
    """
    Get similar users for target user (userId).

    Function will return Pearson Correlation values for users that
    - have rated more than 'moviesInCommonMinimum' identical items with the target user (userId)
    - and have Pearson Correlation value higher that the 'correlationThreshold'.
    """
    targetUserRatings = ratings[ratings['userId'] == userId]

    # get subset of ratings, that only include movies that the target user (userId) has also rated
    userCondition = ratings['userId'] != userId
    movieCondition = ratings['movieId'].isin(targetUserRatings['movieId'].tolist())
    ratingSubset = ratings[userCondition & movieCondition]

    # filter users that do not have rated more than 'moviesInCommonMinimum' identical movies
    ratingSubsetFiltered = ratingSubset[ratingSubset['userId'].map(ratingSubset['userId'].value_counts()) > moviesInCommonMinimum]
   
    # group by users
    ratingSubsetFiltered = ratingSubsetFiltered.groupby(['userId'])

    # calculate Pearson correlation values
    correlations = pearson_correlations(targetUserRatings, ratingSubsetFiltered)

    # filter correlation values that are NOT higher than the threshold
    correlationThresholdCondition = correlations['PearsonCorrelation'] > correlationThreshold
    correlations = correlations[correlationThresholdCondition]

    # sort
    correlations.sort_values(by='PearsonCorrelation', ascending=False, inplace=True)

    return correlations

def pearson_correlations(targetUserRatings, ratingSubsetFiltered):
    """
    Calculate Pearson Correlation value between the target user and candidate users.

    Returns a dataframe with columns PearsonCorrelation and userId, with correlation values higher than 'correlationThreshold'.
    """
    pearsonCorrelationDict = {}
    targetUserRatingsAverage = targetUserRatings['rating'].mean()

    # calculate Pearson Correlation value for each candidate user one by one
    for candidateUserId, candidateUserRatings in ratingSubsetFiltered:

        candidateUserRatingsAverage = candidateUserRatings['rating'].mean()

        # merge
        merged = targetUserRatings.merge(candidateUserRatings, on='movieId', suffixes=('_target', '_candidate'))  

        # add temporary columns that will be used for calculating the Pearson correlation value
        merged['temp_target'] = merged['rating_target'] - targetUserRatingsAverage
        merged['temp2_target'] = (merged['rating_target'] - targetUserRatingsAverage) ** 2
        merged['temp_candidate'] = merged['rating_candidate'] - candidateUserRatingsAverage
        merged['temp2_candidate'] = (merged['rating_candidate'] - candidateUserRatingsAverage) ** 2
        merged['temp'] = merged['temp_target'] * merged['temp_candidate']

        # and calculate the Pearson correlation value
        # if either part of denominator is 0, the correlation value is 0
        correlationValue = merged['temp'].sum() / ( (merged['temp2_target'].sum() ** 0.5) * (merged['temp2_candidate'].sum() ** 0.5) ) if merged['temp2_target'].sum() != 0 and merged['temp2_candidate'].sum() != 0 else 0
        correlationValue = round(correlationValue, 3)
        
        pearsonCorrelationDict[candidateUserId] = correlationValue
        
    # create a correlation dataframe
    correlations = pd.DataFrame.from_dict(pearsonCorrelationDict, orient='index')
    correlations.columns = ['PearsonCorrelation']
    correlations['userId'] = correlations.index
    correlations.index = range(len(correlations))

    # not sorted
    return correlations

def calculate_group_similarity_matrix(users, df_ratings):
    """
    Calculate similarity matrix between users.
    """
    df_group_similarity = pd.DataFrame(index=users, columns=users)

    for i in range(0,len(users) - 1):
        targetUserRatings = df_ratings[df_ratings['userId'] == users[i]]

        # get subset of ratings, that only include movies that the target user (userId) has also rated
        userCondition = df_ratings['userId'].isin(users[(i + 1):])
        movieCondition = df_ratings['movieId'].isin(targetUserRatings['movieId'].tolist())
        ratingSubset = df_ratings[userCondition & movieCondition]
   
        # group by users
        ratingSubset = ratingSubset.groupby(['userId'])

        # calculate Pearson correlation values
        correlations = pearson_correlations(targetUserRatings, ratingSubset)

        # modify index and column
        correlations.set_index('userId', inplace=True)
        correlations.rename(columns={'PearsonCorrelation': users[i]}, inplace=True)

        # fill null values in result dataframe with correlation values from correlations dataframe
        df_group_similarity = df_group_similarity.combine_first(correlations)

    df_group_similarity = df_group_similarity.combine_first(df_group_similarity.T)

    return df_group_similarity



