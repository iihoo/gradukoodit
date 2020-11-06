import pandas as pd
import numpy as np
import random

import correlations
import calculations

# just for wide printouts
pd.set_option('display.expand_frame_repr', False)

NUMBER_OF_USERS = 3
CORRELATION_THRESHOLD = 0.7
MOVIES_IN_COMMON_MINIMUM = 6

ratingsDF = pd.read_csv('movielens-small/ratings.csv')
ratingsDF.drop(['timestamp'], axis=1, inplace=True)

# pick random users
allUsers = ratingsDF['userId'].unique().tolist()
users = []
for i in range(0, NUMBER_OF_USERS):
    user = random.choice(allUsers)
    users.append(user)
    allUsers.remove(user)

# calculate individial recommendation lists (list of dataframes)
recommendations = calculations.calculate_recommendations_all(ratingsDF, users, MOVIES_IN_COMMON_MINIMUM, CORRELATION_THRESHOLD)

### Compare sequential hybrid aggregation method and sequential modified average aggregation
# PARAMETERS
# top-k movies
k = 10

# alfa variable for hybrid aggregation method
alfa = 0

# for the first round, initialize satisfaction score = 1, for each user (modified average aggregation)
satisfactionModifiedAggregation = {u:1 for u in users}

# scale ratings to linear scale using original rating scale from ratings data
ratingScale = ratingsDF['rating'].unique()
ratingScale.sort()
# scale of ratings = tuple of (lowest rating, highest rating)
ratingScale = (ratingScale[0], ratingScale[len(ratingScale) - 1])

RECOMMENDATION_ROUNDS = 5
groupSatOHybrid = []
groupDisOHybrid = []
groupSatOModifiedAggregation = []
groupDisOModifiedAggregation = []

groupDisOHybrid2 = []
groupDisOModifiedAggregation2 = []

# simulate recommendation rounds
for i in range(1, RECOMMENDATION_ROUNDS + 1):
    print('\n**********************************************************************************************************************************')
    print(f'ROUND {i}')

    groupListHybrid = calculations.calculate_group_recommendation_list_hybrid(recommendations, alfa)
    groupListModifiedAggregation = calculations.calculate_group_recommendation_list_modified_average_aggregation(recommendations, satisfactionModifiedAggregation, ratingScale)

    # calculate satisfaction scores, use only top-k items in the group recommendation list
    satisfactionHybrid = calculations.calculate_satisfaction(groupListHybrid, users, k)
    satisfactionModifiedAggregation = calculations.calculate_satisfaction(groupListModifiedAggregation, users, k)

    alfa = max(list(satisfactionHybrid.values())) - min(list(satisfactionHybrid.values()))

    groupSatHybrid = sum(satisfactionHybrid.values()) / len(satisfactionHybrid)
    groupSatOHybrid.append(groupSatHybrid)
    groupDisHybrid = max(satisfactionHybrid.values()) - min(satisfactionHybrid.values())
    groupDisOHybrid.append(groupDisHybrid)

    groupSatModifiedAggregation = sum(satisfactionModifiedAggregation.values()) / len(satisfactionModifiedAggregation)
    groupSatOModifiedAggregation.append(groupSatModifiedAggregation)
    groupDisModifiedAggregation = max(satisfactionModifiedAggregation.values()) - min(satisfactionModifiedAggregation.values())
    groupDisOModifiedAggregation.append(groupDisModifiedAggregation)

    # try also the average of all pairwise disagreements as the dissatisfaction measure
    groupDisOHybrid2.append(calculations.calculate_average_of_all_pairwise_differences(satisfactionHybrid))
    groupDisOModifiedAggregation2.append(calculations.calculate_average_of_all_pairwise_differences(satisfactionModifiedAggregation))

    print('Results, hybrid aggregation:')
    print(groupListHybrid)
    print('Results, modified aggregation:')
    print(groupListModifiedAggregation)


    print(f'\nSatisfaction scores after round {i}')
    print('Hybrid:')
    [print(key, round(value, 4)) for key, value in satisfactionHybrid.items()]
    print(f'GroupSatO: {groupSatHybrid:.4f}')
    print(f'GroupDisO: {groupDisHybrid:.4f}')
    
    print('\nModified:')
    [print(key, round(value, 4)) for key, value in satisfactionModifiedAggregation.items()]
    print(f'GroupSatO: {groupSatModifiedAggregation:.4f}')
    print(f'GroupDisO: {groupDisModifiedAggregation:.4f}')

    # remove top-k movies from both group recommendation lists
    moviesToBeRemoved1 = list(groupListHybrid['movieId'][:k])
    moviesToBeRemoved2 = list(groupListModifiedAggregation['movieId'][:k])
    moviesToBeRemoved = moviesToBeRemoved1 + moviesToBeRemoved2
    # remove from the users' recommendation list
    for i in range(0,len(recommendations)):
        condition = ~recommendations[i].movieId.isin(moviesToBeRemoved)
        recommendations[i] = recommendations[i][condition]


groupSatOHybridAverage = sum(groupSatOHybrid) / len(groupSatOHybrid)
groupDisOHybridAverage = sum(groupDisOHybrid) / len(groupDisOHybrid)
groupSatOModifiedAggregationAverage = sum(groupSatOModifiedAggregation) / len(groupSatOModifiedAggregation)
groupDisOModifiedAggregationAverage = sum(groupDisOModifiedAggregation) / len(groupDisOModifiedAggregation)
print(f'\nAfter {RECOMMENDATION_ROUNDS} recommendation rounds')
print('HYBRID METHOD')
print(f'GroupSatO: {groupSatOHybridAverage:.4f}')
print(f'GroupDisO: {groupDisOHybridAverage:.4f}')
print(f'F-score: {calculations.calculate_F_score(groupSatOHybridAverage, groupDisOHybridAverage):4f}')
print(f'satisfaction scores for each roundd')
print(groupSatOHybrid)
print(f'dissatisfaction scores for each round')
print(groupDisOHybrid)
print('MODIFIED AVERAGE AGGREGATION METHOD')
print(f'GroupSatO: {groupSatOModifiedAggregationAverage:.4f}')
print(f'GroupDisO: {groupDisOModifiedAggregationAverage:.4f}')
print(f'F-score: {calculations.calculate_F_score(groupSatOModifiedAggregationAverage, groupDisOModifiedAggregationAverage):4f}')
print(f'satisfaction scores for each roundd')
print(groupSatOModifiedAggregation)
print(f'dissatisfaction scores for each round')
print(groupDisOModifiedAggregation)


groupDisOHybridAverage2 = sum(groupDisOHybrid2) / len(groupDisOHybrid2)
groupDisOModifiedAggregationAverage2 = sum(groupDisOModifiedAggregation2) / len(groupDisOModifiedAggregation2)
print(f'Hybrid: GroupDisO with average of all pairwise disagreements {groupDisOHybridAverage2}')
print(f'Modified aggregation: GroupDisO with average of all pairwise disagreements {groupDisOModifiedAggregationAverage2}')



'''
### sequential hybrid aggregation method
k = 10
alfa = 0

# simulate recommendation rounds
for i in range(1,3):
    print('\n**********************************************************************************************************************************')
    print(f'ROUND {i}, alfa = {alfa}')

    groupList = calculations.calculate_group_recommendation_list_hybrid(recommendations, alfa)

    # calculate satisfaction scores, use only top-k items in the group recommendation list
    satisfaction = calculations.calculate_satisfaction(groupList, users, k)

    alfa = max(list(satisfaction.values())) - min(list(satisfaction.values()))
    
    print('Results:')
    print(groupList)

    print(f'\nSatisfaction scores after round {i}')
    [print(key, round(value, 4) for key, value in satisfaction.items()]
    print(f'--> alfa = {alfa}')

    # remove top-k movies from the grouplist
    moviesToBeRemoved = list(groupList['movieId'][:k])
    # remove from the users' recommendation list
    for i in range(0,len(recommendations)):
        condition = ~recommendations[i].movieId.isin(moviesToBeRemoved)
        recommendations[i] = recommendations[i][condition]
'''

'''
### sequential modified average aggregation
k = 10
# for the first round, initialize satisfaction score = 1, for each user
satisfaction = {u:1 for u in users}

# scale ratings to linear scale using original rating scale from ratings data
ratingScale = ratingsDF['rating'].unique()
ratingScale.sort()
# scale of ratings = tuple of (lowest rating, highest rating)
ratingScale = (ratingScale[0], ratingScale[len(ratingScale) - 1])

# simulate recommendation rounds
for i in range(1,4):
    print('\n**********************************************************************************************************************************')
    print(f'ROUND {i}')

    groupListModifiedAggregation = calculations.calculate_group_recommendation_list_modified_average_aggregation(recommendations, satisfaction, ratingScale)

    # calculate satisfaction scores, use only top-k items in the group recommendation list
    satisfaction = calculations.calculate_satisfaction(groupListModifiedAggregation, users, k)
    
    print('Results:')
    print(groupListModifiedAggregation)

    print(f'\nSatisfaction scores after round {i}')
    [print(key, round(value, 4)) for key, value in satisfaction.items()]

    # remove top-k movies from the grouplist
    moviesToBeRemoved = list(groupListModifiedAggregation['movieId'][:k])
    # remove from the users' recommendation list
    for i in range(0,len(recommendations)):
        condition = ~recommendations[i].movieId.isin(moviesToBeRemoved)
        recommendations[i] = recommendations[i][condition]
'''

