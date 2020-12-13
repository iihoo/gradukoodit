import pandas as pd

from itertools import combinations
import random

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

    Prediction function (prediction for movie 'p' for user 'a'): 
    prediction(a,p) = r_a_average + sum[similarity(a,b)*(r_b_p - r_b_average)] / sum(similarity(a,b)),
        where r_a_average is the averege of ratings for user a
        where r_b_average is the average of ratings for user b
        where r_b_p is the rating of movie p for user b
        where similarity(a,b) is the Pearson Correlation value between users a and b
        where b are all the users, that have rated movie p.


    NOTE prediction function formula is presented in 'formulas' folder.

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

def calculate_group_recommendation_list_hybrid(recommendationListsDict, alfa, satisfactionScores):
    """
    Assembles a group recommendation list from individual users' recommendation lists.

    Parameter alfa = max satisfaction score - min satisfaction score, from the previous round (alfa = 0 in the first round).

    Returns a dataframe (group recommendation list).
    """
    # combine individual recommendation lists
    df_group_recommendation = combine_recommendation_lists(recommendationListsDict)

    # calculate the average score, and add new column
    df_group_recommendation.insert(1, 'average', df_group_recommendation.iloc[:, 1:].mean(axis=1))

    # calculate the least misery score for HYBRID method (by choosing the predicted rating from the least satisfied user), and add new column
    smallestSatisfaction = 1
    smallestSatisfactionId = random.choice(list(satisfactionScores))
    for key in satisfactionScores:
        if satisfactionScores[key] < smallestSatisfaction:
            smallestSatisfaction = satisfactionScores[key]
            smallestSatisfactionId = key
    df_group_recommendation.insert(2, 'least misery', df_group_recommendation[str(smallestSatisfactionId)])

    # calculate the hybrid score, and add new column
    df_group_recommendation.insert(1, 'result', (1 - alfa) * df_group_recommendation['average'] + alfa * df_group_recommendation['least misery'])

    # sort group recommendation list based on the predicted rating for the group, in descending order
    df_group_recommendation.sort_values(by=['result'], ascending=False, inplace=True)

    return df_group_recommendation

def calculate_group_recommendation_list_average_min_disagreement(recommendationListsDict, satisfactionScores):
    """
    Assembles a group recommendation list from individual users' recommendation lists.

    Returns a dataframe (group recommendation list).
    """
    # combine individual recommendation lists
    df_group_recommendation = combine_recommendation_lists(recommendationListsDict)

    # calculate the average score, and add new column
    df_group_recommendation.insert(1, 'average', df_group_recommendation.iloc[:, 1:].mean(axis=1))

    # sort group recommendation list based on the predicted rating for the group, in descending order
    df_group_recommendation.sort_values(by=['average'], ascending=False, inplace=True)

    cols = 2 + len(satisfactionScores)
    df_group_recommendation.insert(len(df_group_recommendation.columns), 'max diff', abs(df_group_recommendation.iloc[:, 2:cols].div(df_group_recommendation.average, axis=0) - 1).max(axis=1))
    df_group_recommendation.insert(len(df_group_recommendation.columns), 'min diff', abs(df_group_recommendation.iloc[:, 2:cols].div(df_group_recommendation.average, axis=0) - 1).min(axis=1))
    df_group_recommendation.insert(len(df_group_recommendation.columns), 'dis', df_group_recommendation['max diff'] - df_group_recommendation['min diff'])

    df_top = df_group_recommendation[:200].copy()
    df_rest = df_group_recommendation[200:]
    # sort group recommendation list based on the predicted rating for the group, in descending order
    df_top.sort_values(by=['dis'], ascending=True, inplace=True)

    df_top = df_top.append(df_rest, ignore_index = True)

    return df_top

def calculate_group_recommendation_list_adjusted_average(recommendationListsDict, satisfactionScores):
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

    for userId in satisfactionScores:
        user = str(userId)

        factor = abs(satisfactionAverage - satisfactionScores[userId])

        df_group_recommendation[user + ', adj.'] = df_group_recommendation[user]

        condition = df_group_recommendation[user] > df_group_recommendation['average']
        
        df_group_recommendation.loc[condition, user + ', adj.'].apply(lambda r: r * (1 + factor) if (satisfactionScores[userId] < satisfactionAverage) else r)
        df_group_recommendation.loc[~condition, user + ', adj.'].apply(lambda r: r * (1 - factor) if (satisfactionScores[userId] < satisfactionAverage) else r)
        
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

def calculate_F_score(groupSatOAverage, groupDisOAverage):
    return 2 * (groupSatOAverage * (1 - groupDisOAverage)) / (groupSatOAverage + (1 - groupDisOAverage))

def calculate_average_of_all_pairwise_differences(satisfactionScoresDict):
    differences = []
    for x, y in combinations(satisfactionScoresDict.keys(), 2):
        differences.append(abs(satisfactionScoresDict[x] - satisfactionScoresDict[y]))
    return (sum(differences) / len(differences))