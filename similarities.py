import math
import pandas as pd
import numpy as np
import time

def similarity_values(ratings, userId, moviesInCommonMinimum):
    """
    Get similar users for target user (userId).

    Function will return Pearson Correlation values for users that
    - have rated more than 'moviesInCommonMinimum' identical items with the target user (userId)
    """
    targetUserRatings = ratings[ratings['userId'] == userId]

    # get subset of ratings, that only include movies that the target user (userId) has also rated
    userCondition = ratings['userId'] != userId
    movieCondition = ratings['movieId'].isin(targetUserRatings['movieId'].tolist())
    ratingSubset = ratings[userCondition & movieCondition]

    # filter users that do not have rated more than 'moviesInCommonMinimum' identical movies
    ratingSubsetFiltered = ratingSubset[ratingSubset['userId'].map(ratingSubset['userId'].value_counts()) >= moviesInCommonMinimum]

    # calculate Pearson correlation values
    correlations = pearson_correlations(targetUserRatings, ratingSubsetFiltered)

    return correlations


def pearson_correlations(targetUserRatings, ratingSubsetFiltered):
    """
    Calculate Pearson Correlation value between the target user and candidate users.

    Returns a dataframe with columns PearsonCorrelation and userId.
    """

    df_temp = ratingSubsetFiltered.merge(targetUserRatings, on='movieId', suffixes=('_candidate', '_target'))
    df_temp2 = df_temp.groupby('userId_candidate')

    # calculate correlations
    df_temp3 = df_temp2[['rating_candidate','rating_target']].corr()

    idx = pd.IndexSlice
    correlations = df_temp3.loc[idx[:, 'rating_candidate'],['rating_target']]
    correlations.reset_index(inplace=True)
    correlations.drop(columns=['level_1'], inplace=True)
    correlations.rename(columns={'rating_target': 'PearsonCorrelation', 'userId_candidate' : 'userId'}, inplace=True)

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

        # calculate Pearson correlation values
        correlations = pearson_correlations(targetUserRatings, ratingSubset)

        # modify index and column
        correlations.set_index('userId', inplace=True)
        correlations.rename(columns={'PearsonCorrelation': users[i]}, inplace=True)

        # fill null values in result dataframe with correlation values from correlations dataframe
        df_group_similarity = df_group_similarity.combine_first(correlations)

    df_group_similarity = df_group_similarity.combine_first(df_group_similarity.T)

    return df_group_similarity



