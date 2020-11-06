import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from itertools import combinations

import correlations

def calculate_recommendations_all(ratingsDF, users, MOVIES_IN_COMMON_MINIMUM, CORRELATION_THRESHOLD):
    """
    Calculates individual recommendation lists for users 'users'.

    Returns a list of individual recommendation lists (list of dataframes).
    """
    # calculate average rating for each user, and rename column
    average = ratingsDF.groupby('userId').mean().rename(columns={'rating':'average rating, user'})['average rating, user']

    # calculate individual recommendation lists and add to a list
    recommendations = []
    for i in range(0, len(users)):
        correlationsDF = correlations.similar_users(ratingsDF, users[i], MOVIES_IN_COMMON_MINIMUM, CORRELATION_THRESHOLD)
        recommendations.append(calculate_recommendations_single(ratingsDF, average, correlationsDF, users[i]))
    return recommendations

def calculate_recommendations_single(ratingsDF, average, correlationsDF, userId):
    """
    Calculates a recommendation list for a user (userId).

    Returns a dataframe (recommendation list).

    Prediction function: 
    pred(a,p) = average(r_a) + sum( sim(a,b)*(r_b,p - average(r_b)) ) / sum(sim(a,b))
    """

    # merge correlation values to ratings 
    df = correlationsDF.merge(
        ratingsDF, left_on='userId', right_on='userId', how='inner')
   
    # merge average ratings to ratings
    df = df.merge(average, left_on='userId', right_on='userId', how='inner')

    # calculate adjusted ratings and add it as a column
    df['adjusted rating'] = df['PearsonCorrelation'] * (df['rating'] - df['average rating, user'])

    # Applies a sum to the topUsers after grouping it up by userId
    # group by movieId and calculate sum columns 'PearsonCorrelation', 'weighted rating'
    tempValuesDF = df.groupby('movieId').sum()[
        ['PearsonCorrelation', 'adjusted rating']]

    # rename columns
    tempValuesDF.columns = ['sum_PearsonCorr', 'sum_adjusted_rating']
    tempValuesDF['sum_adjusted_rating / sum_PearsonCorr'] = tempValuesDF['sum_adjusted_rating'] / tempValuesDF['sum_PearsonCorr']

    # create a recommendation dataframe
    recommendationDF = pd.DataFrame()
    recommendationDF['recommendation score'] = average[userId] + ( tempValuesDF['sum_adjusted_rating'] / tempValuesDF['sum_PearsonCorr'] )
    recommendationDF['movieId'] = tempValuesDF.index
    recommendationDf = recommendationDF.sort_values(by='recommendation score', ascending=False, inplace=True)

    # get scale of ratings
    ratingScale = ratingsDF['rating'].unique()
    ratingScale.sort()
    ratingScale = (ratingScale[0], ratingScale[len(ratingScale) - 1])
    scaler = MinMaxScaler(feature_range=(ratingScale))

    # scale ratings to linear scale using original rating scale from ratings data
    recommendationDF['prediction for user ' + str(userId)] = scaler.fit_transform(recommendationDF['recommendation score'].values.reshape(-1,1))

    return recommendationDF

def calculate_group_recommendation_list(recommendationLists, aggregationMethod):
    """
    Assembles a group recommendation list from individual users' recommendation lists.

    Aggregation method = 'least misery' or 'average'.

    Returns a dataframe (group recommendation list).
    """
    tempGroupRecommendationDF = recommendationLists[0]
    
    for i in range(1, len(recommendationLists)):
        tempGroupRecommendationDF = tempGroupRecommendationDF.merge(recommendationLists[i], left_on='movieId', right_on='movieId', how='outer', suffixes=(i - 1, i))

    columns = [col for col in tempGroupRecommendationDF.columns if 'prediction' in col or col == 'movieId']
    groupRecommendationDF = tempGroupRecommendationDF[columns]

    # remove rows with NaN values
    # in other words: we only consider the movies, that have a predicted score for each user in the group
    groupRecommendationDF = groupRecommendationDF.dropna()

    # calculate the average score, and add new column
    groupRecommendationDF.insert(1, 'average', groupRecommendationDF.iloc[:, 1:].mean(axis=1))
    #groupListSorted = groupRecommendationDF.sort_values(by=['average'], ascending=False)

    # calculate the least misery score, and add new column
    groupRecommendationDF.insert(2, 'least misery', groupRecommendationDF.iloc[:, 2:].min(axis=1))
    #groupListSorted = groupRecommendationDF.sort_values(by=['least misery'], ascending=False)

    groupRecommendationDF.insert(1, 'result', groupRecommendationDF[aggregationMethod])
    groupListSorted = groupRecommendationDF.sort_values(by=['result'], ascending=False)

    return groupListSorted

def calculate_group_recommendation_list_hybrid(recommendationLists, alfa):
    """
    Assembles a group recommendation list from individual users' recommendation lists.

    Returns a dataframe (group recommendation list).
    """
    tempGroupRecommendationDF = recommendationLists[0]
    
    for i in range(1, len(recommendationLists)):
        tempGroupRecommendationDF = tempGroupRecommendationDF.merge(recommendationLists[i], left_on='movieId', right_on='movieId', how='outer', suffixes=(i - 1, i))

    columns = [col for col in tempGroupRecommendationDF.columns if 'prediction' in col or col == 'movieId']
    groupRecommendationDF = tempGroupRecommendationDF[columns]

    # remove rows with NaN values
    # in other words: we only consider the movies, that have a predicted score for each user in the group
    groupRecommendationDF = groupRecommendationDF.dropna()

    # calculate the average score, and add new column
    groupRecommendationDF.insert(1, 'average', groupRecommendationDF.iloc[:, 1:].mean(axis=1))

    # calculate the least misery score, and add new column
    groupRecommendationDF.insert(2, 'least misery', groupRecommendationDF.iloc[:, 2:].min(axis=1))

    # alfa = max satisfaction score - min satisfaction score, from the previous round
    groupRecommendationDF.insert(1, 'result', (1 - alfa) * groupRecommendationDF['average'] + alfa * groupRecommendationDF['least misery'])

    groupListSorted = groupRecommendationDF.sort_values(by=['result'], ascending=False)
    return groupListSorted

def calculate_group_recommendation_list_modified_average_aggregation(recommendationLists, satisfactionScores, ratingScale):
    """
    Assembles a group recommendation list from individual users' recommendation lists.

    Modified average aggregation is used.

    Returns a dataframe (group recommendation list).
    """
    # calculate the weighing factors for each user for modified average aggregation
    satisfactionAverage = sum(satisfactionScores.values()) / len(satisfactionScores)

    factors = {}
    for user in satisfactionScores:
        factors[user] = abs(satisfactionAverage - satisfactionScores[user])
    
    tempGroupRecommendationDF = recommendationLists[0]
    for i in range(1, len(recommendationLists)):
        tempGroupRecommendationDF = tempGroupRecommendationDF.merge(recommendationLists[i], left_on='movieId', right_on='movieId', how='outer', suffixes=(i - 1, i))

    columns = [col for col in tempGroupRecommendationDF.columns if 'prediction' in col or col == 'movieId']
    groupRecommendationDF = tempGroupRecommendationDF[columns]

    # remove rows with NaN values
    # in other words: we only consider the movies, that have a predicted score for each user in the group
    groupRecommendationDF = groupRecommendationDF.dropna()
    
    # calculate the average score, and add new column
    groupRecommendationDF.insert(1, 'average', groupRecommendationDF.iloc[:, 1:].mean(axis=1))

    ## rules for modified averaga aggregation:
    # if r_u > avg_r and sat_u > avg_sat, then w = 1 - f_u
    # if r_u > avg_r and sat_u < avg_sat, then w = 1 + f_u
    # if r_u < avg_r and sat_u > avg_sat, then w = 1 + f_u
    # if r_u < avg_r and sat_u < avg_sat, then w = 1 - f_u
    adjustedRatings = pd.DataFrame(groupRecommendationDF[['movieId', 'average']])
    for user in satisfactionScores:
        adjustedRatings[str(user)] = groupRecommendationDF['prediction for user ' + str(user)]
        adjustedRatings['prediction for user ' + str(user)] = adjustedRatings[str(user)]

        condition = adjustedRatings[str(user)] > adjustedRatings['average']
        
        # condition: r_u > avg_r:
        # r_adj = (1 - f_u) * r OR (1 + f_u) * r
        #adjustedRatings['prediction for user ' + str(user)][condition] = adjustedRatings['prediction for user ' + str(user)][condition].apply(lambda r: r * (1 - factors[user]) if (satisfactionScores[user] > satisfactionAverage) else r * (1 + factors[user]))
        adjustedRatings.loc[condition, 'prediction for user ' + str(user)] = adjustedRatings.loc[condition, 'prediction for user ' + str(user)].apply(lambda r: r * (1 - factors[user]) if (satisfactionScores[user] > satisfactionAverage) else r * (1 + factors[user]))

        # condition: r_u < avg_r
        # r_adj = (1 + f_u) * r OR (1 - f_u) * r
        #adjustedRatings['prediction for user ' + str(user)][~condition] = adjustedRatings['prediction for user ' + str(user)][~condition].apply(lambda r: r * (1 + factors[user]) if (satisfactionScores[user] > satisfactionAverage) else r * (1 - factors[user]))
        adjustedRatings.loc[~condition, 'prediction for user ' + str(user)].apply(lambda r: r * (1 + factors[user]) if (satisfactionScores[user] > satisfactionAverage) else r * (1 - factors[user]))
    
        ### NOTE what if r_u == avg_r OR sat_u == avg_sat??

        #adjustedRatings.drop([str(user)], axis=1, inplace=True)

    # calculate average of the adjusted predictions and add new column
    adjustedRatings.insert(1, 'modified average', adjustedRatings.iloc[:, 2:].mean(axis=1))

    # scale ratings
    scaler = MinMaxScaler(feature_range=(ratingScale))
    adjustedRatings.insert(1, 'modified average, scaled', scaler.fit_transform(adjustedRatings['modified average'].values.reshape(-1,1)))
    
    adjustedRatingsSorted = adjustedRatings.sort_values(by=['modified average, scaled'], ascending=False)
    return adjustedRatingsSorted

def calculate_satisfaction(groupRecommendationList, users, k):
    """
    Calculates satisfaction scores for each user conserning the group recommendation list, top-k movies are considered.

    Returns a dict, where userId is the key, and the corresponding satisfaction score for the user is the corresponding value.

    Satisfaction score is calculated as satisfaction = GroupListSat/UserListSat, 
    where GroupListSat is the sum of predicted scores of the top-k movies in the group recommendation list, for user u, and,
    where UserListSat is the sum of predicted scores of the top-k movies in the single user recommendation list, for user u.
    """
    satisfaction = {}
    for i in range(0, len(users)):
        user = users[i]
        column = [col for col in groupRecommendationList.columns if col == 'prediction for user ' + str(user)]
        predictedScoreSumGroupList = groupRecommendationList[column][:k].sum()
        predictedScoreSumOwnList = groupRecommendationList[column].sort_values(by=column[0], ascending=False)[:k].sum()
        satisfaction[user] = predictedScoreSumGroupList.array[0] / predictedScoreSumOwnList.array[0]
    return satisfaction

def calculate_F_score(groupSatOAverage, groupDisOAverage):
    return 2 * (groupSatOAverage * (1 - groupDisOAverage)) / (groupSatOAverage + (1 - groupDisOAverage))

def calculate_average_of_all_pairwise_differences(satisfactionScoresDict):
    differences = []
    for x, y in combinations(satisfactionScoresDict.keys(), 2):
        differences.append(abs(satisfactionScoresDict[x] - satisfactionScoresDict[y]))
    return (sum(differences) / len(differences))