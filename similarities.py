import math
import pandas as pd
import numpy as np
import warnings

from scipy import stats
from scipy.stats import PearsonRConstantInputWarning

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
    ratingSubsetFiltered = ratingSubset[ratingSubset['userId'].map(ratingSubset['userId'].value_counts()) > moviesInCommonMinimum]
   
    # group by users
    ratingSubsetFiltered = ratingSubsetFiltered.groupby(['userId'])

    # calculate Pearson correlation values
    correlations = pearson_correlations(targetUserRatings, ratingSubsetFiltered)

    return correlations

def pearson_correlations(targetUserRatings, ratingSubsetFiltered):
    """
    Calculate Pearson Correlation value between the target user and candidate users.

    Returns a dataframe with columns PearsonCorrelation and userId.
    """
    pearsonCorrelationDict = {}

    # calculate Pearson Correlation value for each candidate user one by one
    for candidateUserId, candidateUserRatings in ratingSubsetFiltered:

        # merge
        df_merged = targetUserRatings.merge(candidateUserRatings, on='movieId', suffixes=('_target', '_candidate'))  

        # ignore PearsonRConstantInputWarning from scipy.stats.pearsonr()
        with warnings.catch_warnings():
            warnings.filterwarnings(
                action='ignore',
                category=PearsonRConstantInputWarning,
                message='An input array is constant; the correlation coefficent is not defined.')
            # calculate Pearson correlation value for the ratings of target user and candidate user, returns tuple of (pearsonCorrelationCoefficient, p-value)
            corr = stats.pearsonr(df_merged['rating_target'], df_merged['rating_candidate'])

        # correlationValue = 0 if corr is nan
        correlationValue = round(corr[0], 3) if not np.isnan(corr[0]) else 0
        
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



