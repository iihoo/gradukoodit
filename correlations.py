import math
import pandas as pd

def pearson_correlations(targetUserRatings, similarUserCandidates, correlationThreshold):
    """
    Calculate Pearson Correlation value between the target user and candidate users.
    
    Returns a dataframe with columns PearsonCorr and userId.
    """
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

    # finally, remove those users, that have a correlation value lower than threshold
    return correlationsDF[correlationsDF['PearsonCorr'] > correlationThreshold]

def similar_users(ratingsDF, userId, maxSimilarUsers, correlationThreshold):
    """
    Get similar users for target user (userId).
    Function will return max 'n' users.
    """
    #print('\ntarget user: ', userId)
    targetUserRatings = ratingsDF[ratingsDF['userId'] == userId]

    #print('\nsubset of users that have watched the same movies as the target user')
    condition1 = ratingsDF['userId'] != userId
    condition2 = ratingsDF['movieId'].isin(targetUserRatings['movieId'].tolist())
    userSubset = ratingsDF[condition1 & condition2]

    # sort in descending order
    userSubsetSorted = sorted(userSubset.groupby(
        ['userId']), key=lambda x: len(x[1]), reverse=True)

    return pearson_correlations(targetUserRatings, userSubsetSorted[:maxSimilarUsers], correlationThreshold)