import pandas as pd
import numpy as np
import random

import correlations
import calculations

NUMBER_OF_USERS = 3
CORRELATION_THRESHOLD = 0.2
SIMILAR_USERS_MAX = 50

ratingsDF = pd.read_csv('movielens-small/ratings.csv')
ratingsDF.drop(['timestamp'], axis=1, inplace=True)

# pick random users
allUsers = ratingsDF['userId'].unique().tolist()
users = []
for i in range(0, NUMBER_OF_USERS):
    user = random.choice(allUsers)
    users.append(user)
    allUsers.remove(user)

# calculate individual recommendation lists and add to a list
recommendations = []
for i in range(0,len(users)):
    correlationsDF = correlations.similar_users(ratingsDF, users[i], SIMILAR_USERS_MAX, CORRELATION_THRESHOLD)
    recommendations.append(calculations.calculate_recommendations(ratingsDF, correlationsDF, users[i]))

'''
k = 10
alfa = 0

# simulate recommendation rounds
for i in range(1,3):
    print('\n**********************************************************************************************************************************')
    print('ROUND ', i, ', alfa = ', alfa)

    groupList = calculations.calculate_group_recommendation_list_hybrid(recommendations, alfa)

    # calculate satisfaction scores, use only top-k items in the group recommendation list
    satisfaction = calculations.calculate_satisfaction(groupList, users, k)

    alfa = max(list(satisfaction.values())) - min(list(satisfaction.values()))
    
    print('Results:')
    print(groupList)

    print('\nSatisfaction scores after round ', i )
    [print(key, value) for key, value in satisfaction.items()]
    print('--> alfa = ', alfa)

    # remove top-k movies from the grouplist
    moviesToBeRemoved = list(groupList['movieId'][:k])
    # remove from the users' recommendation list
    for i in range(0,len(recommendations)):
        condition = ~recommendations[i].movieId.isin(moviesToBeRemoved)
        recommendations[i] = recommendations[i][condition]
'''

# for the first round, initialize satisfaction score = 1, for each user
satisfaction = {u:1 for u in users}
groupListModifiedAggregation = calculations.calculate_group_recommendation_list_modified_average_aggregation(recommendations, satisfaction)

