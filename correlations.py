import math
import pandas as pd

def pearson_correlations(targetUserRatings, similarUserCandidates, moviesInCommonMinimum, correlationThreshold):
    """
    Calculate Pearson Correlation value between the target user and candidate users.
    
    Similar users are those users that have rated more than 'moviesInCommonMinimum' movies 
    and have a correlation value higher than 'correlationThreshold'.

    Returns a dataframe with columns PearsonCorr and userId.
    """
    pearsonCorrelationDict = {}

    for userId, candidateUserRatings in similarUserCandidates:
        candidateUserRatings = candidateUserRatings.sort_values(by='movieId')
        targetUserRatings = targetUserRatings.sort_values(by='movieId')

        candidateUserRatingsAverage = candidateUserRatings['rating'].mean()
        targetUserRatingsAverage = targetUserRatings['rating'].mean()

        # merge, but reorder columns first
        targetUserRatings = targetUserRatings.reindex(columns=['movieId','userId','rating'])
        
        merged = targetUserRatings.merge(candidateUserRatings, on='movieId', suffixes=('_target', '_candidate'))

        # if the number of identical rated movies is not enough skip this user candidate
        if (merged.shape[0] <= correlationThreshold):
            continue        

        # add temporary columns that will be used for calculating the Pearson correlation value
        merged['temp_target'] = merged['rating_target'] - targetUserRatingsAverage
        merged['temp2_target'] = (merged['rating_target'] - targetUserRatingsAverage) ** 2
        merged['temp_candidate'] = merged['rating_candidate'] - candidateUserRatingsAverage
        merged['temp2_candidate'] = (merged['rating_candidate'] - candidateUserRatingsAverage) ** 2
        merged['temp'] = merged['temp_target'] * merged['temp_candidate']

        # and calculate the Pearson correlation value
        # if either part of denominator is 0, the correlation value is 0
        correlationValue = merged['temp'].sum() / ( (merged['temp2_target'].sum() ** 0.5) * (merged['temp2_candidate'].sum() ** 0.5) ) if merged['temp2_target'].sum() != 0 and merged['temp2_candidate'].sum() != 0 else 0

        # save to dict
        pearsonCorrelationDict[userId] = correlationValue
        
    # create a correlation dataframe and sort according to the correlation value (descending)
    correlationsDF = pd.DataFrame.from_dict(
        pearsonCorrelationDict, orient='index')
    correlationsDF.columns = ['PearsonCorr']
    correlationsDF['userId'] = correlationsDF.index
    correlationsDF.index = range(len(correlationsDF))
    # NOTE: is sorting necessary?
    correlationsDF.sort_values(by='PearsonCorr', ascending=False, inplace=True)

    print('CORRELATIONS, within threshold')
    print(correlationsDF[correlationsDF['PearsonCorr'] > correlationThreshold])

    # finally, only return those users, that have a correlation value higher than the threshold
    return correlationsDF[correlationsDF['PearsonCorr'] > correlationThreshold]

def similar_users(ratingsDF, userId, moviesInCommonMinimum, correlationThreshold):
    """
    Get similar users for target user (userId).

    Function will return users that have rated more than 'moviesInCommonMinimum' identical items with the target user (userId).
    """
    targetUserRatings = ratingsDF[ratingsDF['userId'] == userId]

    condition1 = ratingsDF['userId'] != userId
    condition2 = ratingsDF['movieId'].isin(targetUserRatings['movieId'].tolist())
    userSubset = ratingsDF[condition1 & condition2]

    # NOTE: is sorting necessary?
    # sort in descending order
    #userSubsetSorted = sorted(userSubset.groupby(
    #    ['userId']), key=lambda x: len(x[1]), reverse=True)

    userSubset = userSubset.groupby(['userId'])

    correlations = pearson_correlations(targetUserRatings, userSubset, moviesInCommonMinimum, correlationThreshold)
    return correlations