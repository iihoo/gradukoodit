import pandas as pd

from itertools import combinations

import similarities

def calculate_recommendations_all(ratings, scaler, users, MOVIES_IN_COMMON_MINIMUM, CORRELATION_THRESHOLD):
    """
    Calculates individual recommendation lists for users 'users'.

    Returns individual recommendation lists as a dict, where userId is the dict key.
    """
    # calculate average rating for each user, and rename column
    average = ratings.groupby('userId').mean().rename(columns={'rating':'average rating, user'})['average rating, user']

    # calculate individual recommendation lists and add to a dict
    recommendations = {}
    for i in range(0, len(users)):
        correlations = similarities.similarity_values(ratings, users[i], MOVIES_IN_COMMON_MINIMUM)
        # filter correlation values that are NOT higher than the threshold
        correlationThresholdCondition = correlations['PearsonCorrelation'] > CORRELATION_THRESHOLD
        correlations = correlations[correlationThresholdCondition]

        # sort
        correlations.sort_values(by='PearsonCorrelation', ascending=False, inplace=True)
        recommendations[users[i]] = calculate_recommendations_single(ratings, scaler, average, correlations, users[i])
    return recommendations

def calculate_recommendations_single(ratings, scaler, average, correlations, userId):
    """
    Calculates a recommendation list for a user (userId).

    Prediction function: 
    prediction(a,p) = average(r_a) + sum( similarity(a,b)*(r_b,p - average(r_b)) ) / sum(similarity(a,b))

    Parameter 'scaler' is a MinMaxScaler that is used to scale ratings according to the original rating data.

    Returns a dataframe (recommendation list).
    """
    # merge correlation values to ratings 
    df = correlations.merge(ratings, left_on='userId', right_on='userId', how='inner')
   
    # merge average ratings to ratings
    df = df.merge(average, left_on='userId', right_on='userId', how='inner')

    # calculate adjusted ratings and add it as a column
    df['adjusted rating'] = df['PearsonCorrelation'] * (df['rating'] - df['average rating, user'])

    # Create a temporary dataframe, group by movieId and calculate sum of columns 'PearsonCorrelation', 'weighted rating'
    df_temp = df.groupby('movieId').sum()[['PearsonCorrelation', 'adjusted rating']]

    # rename columns
    df_temp.columns = ['sum_PearsonCorr', 'sum_adjusted_rating']

    # calculate and add a column for sum of adjusted ratings divided by sum of correlation values
    df_temp['sum_adjusted_rating / sum_PearsonCorr'] = df_temp['sum_adjusted_rating'] / df_temp['sum_PearsonCorr']

    # create a recommendation dataframe
    df_recommendation = pd.DataFrame()
    df_recommendation['recommendation score'] = average[userId] + df_temp['sum_adjusted_rating / sum_PearsonCorr']
    df_recommendation['movieId'] = df_temp.index

    # scale ratings to linear scale using original rating scale from ratings data
    df_recommendation['prediction'] = scaler.fit_transform(df_recommendation['recommendation score'].values.reshape(-1,1))

    # only return the scaled rating
    df_recommendation.drop(columns=['recommendation score'], inplace=True)

    # finally, sort according to rating
    df_recommendation.sort_values(by=['prediction'], ascending=False, inplace=True)

    return df_recommendation

def calculate_group_recommendation_list_hybrid(recommendationListsDict, alfa):
    """
    Assembles a group recommendation list from individual users' recommendation lists.

    Parameter alfa = max satisfaction score - min satisfaction score, from the previous round (alfa = 0 in the first round).

    Returns a dataframe (group recommendation list).
    """
    # combine individual recommendation lists
    df_group_recommendation = combine_recommendation_lists(recommendationListsDict)

    # calculate the average score, and add new column
    df_group_recommendation.insert(1, 'average', df_group_recommendation.iloc[:, 1:].mean(axis=1))

    # calculate the least misery score, and add new column
    df_group_recommendation.insert(2, 'least misery', df_group_recommendation.iloc[:, 2:].min(axis=1))

    # calculate the hybrid score, and add new column
    df_group_recommendation.insert(1, 'result', (1 - alfa) * df_group_recommendation['average'] + alfa * df_group_recommendation['least misery'])

    # sort group recommendation list based on the predicted rating for the group, in descending order
    df_group_recommendation.sort_values(by=['result'], ascending=False, inplace=True)

    return df_group_recommendation

def calculate_group_recommendation_list_modified_average_aggregation(recommendationListsDict, satisfactionScores):
    """
    Assembles a group recommendation list from individual users' recommendation lists.

    Modified average aggregation is used.

    Returns a dataframe (group recommendation list).
    """    
    # combine individual recommendation lists
    df_group_recommendation = combine_recommendation_lists(recommendationListsDict)

    # calculate the average score, and add new column
    df_group_recommendation.insert(1, 'average', df_group_recommendation.iloc[:, 1:].mean(axis=1))

    # calculate the average satisfaction from previous round
    satisfactionAverage = sum(satisfactionScores.values()) / len(satisfactionScores)

    ## rules for modified average aggregation:
    # if r_u > avg_r and sat_u > avg_sat, then w = 1 - f_u
    # if r_u > avg_r and sat_u < avg_sat, then w = 1 + f_u
    # if r_u < avg_r and sat_u > avg_sat, then w = 1 + f_u
    # if r_u < avg_r and sat_u < avg_sat, then w = 1 - f_u
    for userId in satisfactionScores:
        user = str(userId)
        # calculate the difference between user's satisfaction compared to the average satisfaction
        factor = abs(satisfactionAverage - satisfactionScores[userId])

        df_group_recommendation[user + ', adj.'] = df_group_recommendation[user]

        condition = df_group_recommendation[user] > df_group_recommendation['average']
        
        # condition: r_u > avg_r: r_adj = (1 - f_u) * r OR (1 + f_u) * r
        df_group_recommendation.loc[condition, user + ', adj.'] = df_group_recommendation.loc[condition, user + ', adj.'].apply(lambda r: r * (1 - factor) if (satisfactionScores[userId] > satisfactionAverage) else r * (1 + factor))

        # condition: r_u < avg_r: r_adj = (1 + f_u) * r OR (1 - f_u) * r
        df_group_recommendation.loc[~condition, user + ', adj.'].apply(lambda r: r * (1 + factor) if (satisfactionScores[userId] > satisfactionAverage) else r * (1 - factor))

        ### NOTE what if r_u == avg_r OR sat_u == avg_sat??

    # calculate average of the adjusted predictions and add a new column
    adjusted = [column for column in df_group_recommendation.columns if 'adj.' in column]
    df_group_recommendation.insert(1, 'result', df_group_recommendation[adjusted].mean(axis=1)) 
    
    # sort ratings
    df_group_recommendation.sort_values(by='result', ascending=False, inplace=True)
    return df_group_recommendation

def combine_recommendation_lists(recommendationListsDict):
    """
    Combine individual recommendation lists.
    """
    # create a group recommendation dataframe by adding individual recommendation lists one by one
    users = list(recommendationListsDict.keys())
    df_combined = recommendationListsDict[users[0]].copy()
    df_combined.rename(columns={'prediction': str(users[0])}, inplace=True)
    for user in users[1:]:
        df_combined = df_combined.merge(recommendationListsDict[user], left_on='movieId', right_on='movieId', how='outer')
        df_combined.rename(columns={'prediction': str(user)}, inplace=True)

    # remove rows with NaN values: only keep the movies, that have a predicted score for each user in the group
    df_combined = df_combined.dropna()
    return df_combined

def calculate_satisfaction(df_group_recommendation, recommendations, k):
    """
    Calculates satisfaction scores for each user conserning the group recommendation list, top-k movies are considered.

    Returns a dict, where userId is the key, and the corresponding satisfaction score for the user is the corresponding value.
    
    Satisfaction score is calculated as satisfaction = GroupListSatisfaction/UserListSatisfaction, 
    where GroupListSatisfaction is the sum of predicted scores of the top-k movies in the group recommendation list, for user u, and,
    where UserListSatisfaction is the sum of predicted scores of the top-k movies in the individual recommendation list, for user u.
    """
    satisfaction = {}
    for user in recommendations.keys():
        groupListSatisfaction = df_group_recommendation[str(user)][:k].sum()
        userListSatisfaction = recommendations[user]['prediction'][:k].sum()
        satisfaction[user] = groupListSatisfaction / userListSatisfaction
    return satisfaction

def remove_movies(recommendations, listOfGroupRecommendationLists, k):
    """
    Remove movies (from individual users recommendation lists) that have already been recommended (top-k) in the group recommendation lists.

    This method works also with multiple group recommendation methods (when comparing methods).

    Returns modified individual recommendation lists (dict), where userId is the dict key, and individual recommendation list for that user is the dict value.
    """
    moviesToBeRemoved = set(listOfGroupRecommendationLists[0]['movieId'][:k])
    for i in range(1, len(listOfGroupRecommendationLists)):
        moviesToBeRemoved.update(listOfGroupRecommendationLists[i]['movieId'][:k])

    # remove from the users' recommendation list
    for user in recommendations.keys():
        condition = ~recommendations[user].movieId.isin(moviesToBeRemoved)
        recommendations[user] = recommendations[user][condition]

    return recommendations

def calculate_F_score(groupSatOAverage, groupDisOAverage):
    return 2 * (groupSatOAverage * (1 - groupDisOAverage)) / (groupSatOAverage + (1 - groupDisOAverage))

def calculate_average_of_all_pairwise_differences(satisfactionScoresDict):
    differences = []
    for x, y in combinations(satisfactionScoresDict.keys(), 2):
        differences.append(abs(satisfactionScoresDict[x] - satisfactionScoresDict[y]))
    return (sum(differences) / len(differences))