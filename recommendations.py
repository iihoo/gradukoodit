import pandas as pd
import numpy as np
import random

from sklearn.preprocessing import MinMaxScaler

import calculations
import visualization

# just for wide printouts
pd.set_option('display.expand_frame_repr', False)

NUMBER_OF_USERS = 3
CORRELATION_THRESHOLD = 0.7
MOVIES_IN_COMMON_MINIMUM = 6
RECOMMENDATION_ROUNDS = 5
INITIAL_DATA_CHUNK_SIZE = 100000


# get initial data chunk 
initialRatingsDataChunk = pd.read_csv('movielens-25m/ratings.csv', usecols=['userId', 'movieId', 'rating'], chunksize=INITIAL_DATA_CHUNK_SIZE)
df_ratings_initial_chunk = initialRatingsDataChunk.get_chunk()

# get scale of ratings
ratingScale = df_ratings_initial_chunk['rating'].unique()
ratingScale.sort()
ratingScale = (ratingScale[0], ratingScale[len(ratingScale) - 1])
scaler = MinMaxScaler(feature_range=(ratingScale))

# pick random users
allUsers = df_ratings_initial_chunk['userId'].unique().tolist()
users = []
for i in range(0, NUMBER_OF_USERS):
    user = random.choice(allUsers)
    users.append(user)
    allUsers.remove(user)

# calculate individial recommendation lists (a dict, where userId is the key and the recommendation list for that user is the dict value)
recommendations = calculations.calculate_recommendations_all(df_ratings_initial_chunk, scaler, users, MOVIES_IN_COMMON_MINIMUM, CORRELATION_THRESHOLD)

### Compare sequential hybrid aggregation method and sequential modified average aggregation
# top-k movies
k = 10

# alfa variable for hybrid aggregation method
alfa = 0

# for the first round, initialize satisfaction score = 1, for each user (modified average aggregation)
satisfactionModifiedAggregation = {u:1 for u in users}

# create a DataFrame for satisfaction & dissatisfaction scores
df_scores = pd.DataFrame(columns=['GroupSatO:HYBRID', 'GroupSatO:MODIF.AGGR.', 'GroupDisO:HYBRID', 'GroupDisO:MODIF.AGGR.'])

# simulate recommendation rounds
for i in range(1, RECOMMENDATION_ROUNDS + 1):
    print('\n**********************************************************************************************************************************')
    print(f'ROUND {i}')

    groupListHybrid = calculations.calculate_group_recommendation_list_hybrid(users, recommendations, alfa)
    groupListModifiedAggregation = calculations.calculate_group_recommendation_list_modified_average_aggregation(users, recommendations, satisfactionModifiedAggregation, scaler)

    # calculate satisfaction scores, use only top-k items in the group recommendation list
    satisfactionHybrid = calculations.calculate_satisfaction(groupListHybrid, users, k)
    satisfactionModifiedAggregation = calculations.calculate_satisfaction(groupListModifiedAggregation, users, k)

    # try to make satisfaction calculations more efficient
    # NOTE try to calculate 
    #satisfactionHybridTest = calculations.calculate_satisfaction_test(groupListHybrid, users, recommendations, k)
    #satisfactionModifiedAggregationTest = calculations.calculate_satisfaction_test(groupListModifiedAggregation, recommendations, users, k)

    alfa = max(list(satisfactionHybrid.values())) - min(list(satisfactionHybrid.values()))

    # calculate the average satisfaction scores from this round
    groupSatHybrid = sum(satisfactionHybrid.values()) / len(satisfactionHybrid)
    groupSatModifiedAggregation = sum(satisfactionModifiedAggregation.values()) / len(satisfactionModifiedAggregation)

    # calculate the dissatisfaction scores from this round
    groupDisHybrid = max(satisfactionHybrid.values()) - min(satisfactionHybrid.values())
    groupDisModifiedAggregation = max(satisfactionModifiedAggregation.values()) - min(satisfactionModifiedAggregation.values())

    # add to results dataframe
    df_scores.loc[i] = [groupSatHybrid, groupSatModifiedAggregation, groupDisHybrid, groupDisModifiedAggregation]

    # top-k results
    print('\nResults, hybrid aggregation:')
    print(groupListHybrid[:k])
    print('\nResults, modified aggregation:')
    print(groupListModifiedAggregation[:k])

    # remove top-k movies from both group recommendation lists
    moviesToBeRemoved1 = list(groupListHybrid['movieId'][:k])
    moviesToBeRemoved2 = list(groupListModifiedAggregation['movieId'][:k])
    moviesToBeRemoved = moviesToBeRemoved1 + moviesToBeRemoved2
    # remove from the users' recommendation list
    for i in range(0,len(users)):
        condition = ~recommendations[users[i]].movieId.isin(moviesToBeRemoved)
        recommendations[i] = recommendations[users[i]][condition]

# calculate average of the average of group satisfaction scores
groupSatOHybridAverage = df_scores['GroupSatO:HYBRID'].mean()
groupSatOModifiedAggregationAverage = df_scores['GroupSatO:MODIF.AGGR.'].mean()

# calculate average of the group dissatisfaction scores
groupDisOHybridAverage = df_scores['GroupDisO:HYBRID'].mean()
groupDisOModifiedAggregationAverage = df_scores['GroupDisO:MODIF.AGGR.'].mean()

print(f'\nAfter {RECOMMENDATION_ROUNDS} recommendation rounds')

print('Scores:')
print(df_scores)

print('\nHYBRID METHOD')
print(f'GroupSatO: {groupSatOHybridAverage:.4f}')
print(f'GroupDisO: {groupDisOHybridAverage:.4f}')
print(f'F-score: {calculations.calculate_F_score(groupSatOHybridAverage, groupDisOHybridAverage):4f}')
print(f'satisfaction scores for each round')
print(df_scores['GroupSatO:HYBRID'].to_numpy())
print(f'dissatisfaction scores for each round')
print(df_scores['GroupDisO:HYBRID'].to_numpy())

print('\nMODIFIED AVERAGE AGGREGATION METHOD')
print(f'GroupSatO: {groupSatOModifiedAggregationAverage:.4f}')
print(f'GroupDisO: {groupDisOModifiedAggregationAverage:.4f}')
print(f'F-score: {calculations.calculate_F_score(groupSatOModifiedAggregationAverage, groupDisOModifiedAggregationAverage):4f}')
print(f'satisfaction scores for each round')
print(df_scores['GroupSatO:MODIF.AGGR.'].to_numpy())
print(f'dissatisfaction scores for each round')
print(df_scores['GroupDisO:MODIF.AGGR.'].to_numpy())

### PLOT RESULTS
visualization.plot_satisfaction_dissatisfaction(df_scores)
