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
    
        print('target:')
        print(targetUserRatings)
        print('candidate:')
        print(candidateUserRatings)

        # merge, but reorder columns first
        targetUserRatings = targetUserRatings.reindex(columns=['movieId','userId','rating'])
        
        merged = targetUserRatings.merge(candidateUserRatings, on='movieId', suffixes=('_target', '_candidate'))
        print('merged:')
        print(merged)

        # if the number of identical rated movies is not enough skip this user candidate
        if (merged.shape[0] <= correlationThreshold):
            continue        

        # add temporary columns that will be used for calculating the Pearson correlation value
        merged['temp_target'] = merged['rating_target'] - targetUserRatingsAverage
        merged['temp2_target'] = (merged['rating_target'] - targetUserRatingsAverage) ** 2
        merged['temp_candidate'] = merged['rating_candidate'] - candidateUserRatingsAverage
        merged['temp2_candidate'] = (merged['rating_candidate'] - candidateUserRatingsAverage) ** 2
        merged['temp'] = merged['temp_target'] * merged['temp_candidate']
        print('merged, with temp values:')
        print(merged)

        # and calculate the Pearson correlation value
        correlationValue = merged['temp'].sum() / ( (merged['temp2_target'].sum() ** 0.5) * (merged('temp2_candidate').sum() ** 0.5) )
        print('correlation')
        print(correlationValue)

        numberOfRatings = len(candidateUserRatings)
        moviesInCommon = targetUserRatings[targetUserRatings['movieId'].isin(
            candidateUserRatings['movieId'].tolist())]
        ratingsOfMoviesInCommonByTargetUser = moviesInCommon['rating'].tolist()
        ratingsOfOtherUser = candidateUserRatings['rating'].tolist()

        Sxy = sum(i*j for i, j in zip(ratingsOfMoviesInCommonByTargetUser, ratingsOfOtherUser)) - \
            sum(ratingsOfMoviesInCommonByTargetUser) * \
            sum(ratingsOfOtherUser)/float(numberOfRatings)
        Sxx = sum([i**2 for i in ratingsOfMoviesInCommonByTargetUser]) - \
            pow(sum(ratingsOfMoviesInCommonByTargetUser), 2) / \
            float(numberOfRatings)
        Syy = sum([i**2 for i in ratingsOfOtherUser]) - \
            pow(sum(ratingsOfOtherUser), 2)/float(numberOfRatings)

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
    # NOTE: is sorting necessary?
    correlationsDF.sort_values(by='PearsonCorr', ascending=False, inplace=True)

    print(correlationsDF)

    # finally, remove those users, that have a correlation value lower than threshold
    return correlationsDF[correlationsDF['PearsonCorr'] > correlationThreshold]

def similar_users(ratingsDF, userId, moviesInCommonMinimum, correlationThreshold):
    """
    Get similar users for target user (userId).
    Function will return n users (n = maxSimilarUsers).
    """
    targetUserRatings = ratingsDF[ratingsDF['userId'] == userId]

    condition1 = ratingsDF['userId'] != userId
    condition2 = ratingsDF['movieId'].isin(targetUserRatings['movieId'].tolist())
    userSubset = ratingsDF[condition1 & condition2]

    # NOTE: is sorting necessary?
    # sort in descending order
    #userSubsetSorted = sorted(userSubset.groupby(
    #    ['userId']), key=lambda x: len(x[1]), reverse=True)

    # NOTE consider all candidates
    userSubset = userSubset[:10].groupby(['userId'])

    correlations = pearson_correlations(targetUserRatings, userSubset, moviesInCommonMinimum, correlationThreshold)
    return correlations