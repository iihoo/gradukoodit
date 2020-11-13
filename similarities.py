import math
import pandas as pd

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
    correlations = pearson_correlations(targetUserRatings, ratingSubsetFiltered, correlationThreshold)

    return correlations

def pearson_correlations(targetUserRatings, ratingSubsetFiltered, correlationThreshold):
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

        # save to dict, if correlation value is higher than the threshold 
        if correlationValue > correlationThreshold:
            pearsonCorrelationDict[candidateUserId] = correlationValue
        
    # create a correlation dataframe
    correlations = pd.DataFrame.from_dict(pearsonCorrelationDict, orient='index')
    correlations.columns = ['PearsonCorrelation']
    correlations['userId'] = correlations.index
    correlations.index = range(len(correlations))

    # sort
    correlations.sort_values(by='PearsonCorrelation', ascending=False, inplace=True)

    return correlations