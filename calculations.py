import pandas as pd

from itertools import combinations

import similarities

def calculate_recommendations_all(ratings, scaler, users, MOVIES_IN_COMMON_MINIMUM, CORRELATION_THRESHOLD):
    """
    Calculates individual recommendation lists for users 'users'.

    Returns a list of individual recommendation lists (list of dataframes).
    """
    # calculate average rating for each user, and rename column
    average = ratings.groupby('userId').mean().rename(columns={'rating':'average rating, user'})['average rating, user']

    # calculate individual recommendation lists and add to a list
    recommendations = []
    for i in range(0, len(users)):
        correlations = similarities.similar_users(ratings, users[i], MOVIES_IN_COMMON_MINIMUM, CORRELATION_THRESHOLD)
        recommendations.append(calculate_recommendations_single(ratings, scaler, average, correlations, users[i]))
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
    df_recommendation['recommendation score'] = average[userId] + ( df_temp['sum_adjusted_rating'] / df_temp['sum_PearsonCorr'] )
    df_recommendation['movieId'] = df_temp.index

    # scale ratings to linear scale using original rating scale from ratings data
    df_recommendation['prediction for user ' + str(userId)] = scaler.fit_transform(df_recommendation['recommendation score'].values.reshape(-1,1))

    return df_recommendation

def calculate_group_recommendation_list_hybrid(recommendationLists, alfa):
    """
    Assembles a group recommendation list from individual users' recommendation lists.

    Parameter alfa = max satisfaction score - min satisfaction score, from the previous round (alfa = 0 in the first round).

    Returns a dataframe (group recommendation list).
    """
    # create a temporary dataframe and add individual recommendation lists one by one
    df_temp = recommendationLists[0]
    for i in range(1, len(recommendationLists)):
        df_temp = df_temp.merge(recommendationLists[i], left_on='movieId', right_on='movieId', how='outer', suffixes=(i - 1, i))

    columns = [col for col in df_temp.columns if 'prediction' in col or col == 'movieId']
    df_group_recommendation = df_temp[columns]

    # remove rows with NaN values: only keep the movies, that have a predicted score for each user in the group
    df_group_recommendation = df_group_recommendation.dropna()

    # calculate the average score, and add new column
    df_group_recommendation.insert(1, 'average', df_group_recommendation.iloc[:, 1:].mean(axis=1))

    # calculate the least misery score, and add new column
    df_group_recommendation.insert(2, 'least misery', df_group_recommendation.iloc[:, 2:].min(axis=1))

    # calculate the hybrid score, and add new column
    df_group_recommendation.insert(1, 'result', (1 - alfa) * df_group_recommendation['average'] + alfa * df_group_recommendation['least misery'])

    # sort group recommendation list based on the predicted rating for the group, in descending order
    df_group_recommendation.sort_values(by=['result'], ascending=False, inplace=True)

    return df_group_recommendation

def calculate_group_recommendation_list_modified_average_aggregation(recommendationLists, satisfactionScores, scaler):
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
    
    # create a temporary dataframe and add individual recommendation lists one by one
    df_temp = recommendationLists[0]
    for i in range(1, len(recommendationLists)):
        df_temp = df_temp.merge(recommendationLists[i], left_on='movieId', right_on='movieId', how='outer', suffixes=(i - 1, i))

    columns = [col for col in df_temp.columns if 'prediction' in col or col == 'movieId']
    df_group_recommendation = df_temp[columns]

    # remove rows with NaN values: only keep the movies, that have a predicted score for each user in the group
    df_group_recommendation = df_group_recommendation.dropna()
    
    # calculate the average score, and add new column
    df_group_recommendation.insert(1, 'average', df_group_recommendation.iloc[:, 1:].mean(axis=1))

    ## rules for modified averaga aggregation:
    # if r_u > avg_r and sat_u > avg_sat, then w = 1 - f_u
    # if r_u > avg_r and sat_u < avg_sat, then w = 1 + f_u
    # if r_u < avg_r and sat_u > avg_sat, then w = 1 + f_u
    # if r_u < avg_r and sat_u < avg_sat, then w = 1 - f_u
    df_adjusted_ratings = pd.DataFrame(df_group_recommendation[['movieId', 'average']])
    for user in satisfactionScores:
        df_adjusted_ratings[str(user)] = df_group_recommendation['prediction for user ' + str(user)]
        df_adjusted_ratings['prediction for user ' + str(user)] = df_adjusted_ratings[str(user)]

        condition = df_adjusted_ratings[str(user)] > df_adjusted_ratings['average']
        
        # condition: r_u > avg_r:
        # r_adj = (1 - f_u) * r OR (1 + f_u) * r
        #df_adjusted_ratings['prediction for user ' + str(user)][condition] = df_adjusted_ratings['prediction for user ' + str(user)][condition].apply(lambda r: r * (1 - factors[user]) if (satisfactionScores[user] > satisfactionAverage) else r * (1 + factors[user]))
        df_adjusted_ratings.loc[condition, 'prediction for user ' + str(user)] = df_adjusted_ratings.loc[condition, 'prediction for user ' + str(user)].apply(lambda r: r * (1 - factors[user]) if (satisfactionScores[user] > satisfactionAverage) else r * (1 + factors[user]))

        # condition: r_u < avg_r
        # r_adj = (1 + f_u) * r OR (1 - f_u) * r
        #df_adjusted_ratings['prediction for user ' + str(user)][~condition] = df_adjusted_ratings['prediction for user ' + str(user)][~condition].apply(lambda r: r * (1 + factors[user]) if (satisfactionScores[user] > satisfactionAverage) else r * (1 - factors[user]))
        df_adjusted_ratings.loc[~condition, 'prediction for user ' + str(user)].apply(lambda r: r * (1 + factors[user]) if (satisfactionScores[user] > satisfactionAverage) else r * (1 - factors[user]))
    
        ### NOTE what if r_u == avg_r OR sat_u == avg_sat??

        #df_adjusted_ratings.drop([str(user)], axis=1, inplace=True)

    # calculate average of the adjusted predictions and add new column
    df_adjusted_ratings.insert(1, 'modified average', df_adjusted_ratings.iloc[:, 2:].mean(axis=1))

    # scale ratings
    df_adjusted_ratings.insert(1, 'modified average, scaled', scaler.fit_transform(df_adjusted_ratings['modified average'].values.reshape(-1,1)))
    
    # sort group recommendation list based on the predicted rating for the group, in descending order
    df_adjusted_ratings.sort_values(by='modified average, scaled', ascending=False, inplace=True)

    return df_adjusted_ratings

def calculate_satisfaction(df_group_recommendation, users, k):
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
        column = [col for col in df_group_recommendation.columns if col == 'prediction for user ' + str(user)]
        predictedScoreSumGroupList = df_group_recommendation[column][:k].sum().array[0]
        predictedScoreSumOwnList = df_group_recommendation[column].sort_values(by=column[0], ascending=False)[:k].sum().array[0]
        satisfaction[user] = predictedScoreSumGroupList / predictedScoreSumOwnList
    return satisfaction

def calculate_F_score(groupSatOAverage, groupDisOAverage):
    return 2 * (groupSatOAverage * (1 - groupDisOAverage)) / (groupSatOAverage + (1 - groupDisOAverage))

def calculate_average_of_all_pairwise_differences(satisfactionScoresDict):
    differences = []
    for x, y in combinations(satisfactionScoresDict.keys(), 2):
        differences.append(abs(satisfactionScoresDict[x] - satisfactionScoresDict[y]))
    return (sum(differences) / len(differences))