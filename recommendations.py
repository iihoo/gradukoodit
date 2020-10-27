import pandas as pd
import numpy as np
import random

from correlations import pearson_correlations, similar_users
from calculations import calculate_recommendations, calculate_group_recommendation_list, calculate_group_recommendation_list_hybrid, calculate_satisfaction

NUMBER_OF_USERS = 5
CORRELATION_THRESHOLD = 0.2
SIMILAR_USERS_MAX = 50

ratingsDF = pd.read_csv('movielens-small/ratings.csv')
ratingsDF.drop(['timestamp'], axis=1, inplace=True)

allUsers = ratingsDF['userId'].unique().tolist()
users = []
# pick random users
for i in range(0, NUMBER_OF_USERS):
    user = random.choice(allUsers)
    users.append(user)
    allUsers.remove(user)

### ROUND 1
# at first round, the average aggregation is implemented (alfa = 0)

# calculate individual recommendation lists and add to a list
recommendations = []
for i in range(0,len(users)):
    correlationsDF = similar_users(ratingsDF, users[i], SIMILAR_USERS_MAX, CORRELATION_THRESHOLD)
    recommendations.append(calculate_recommendations(ratingsDF, correlationsDF, users[i]))

groupList = calculate_group_recommendation_list_hybrid(recommendations, 0)

# calculate satisfaction scores, use only top-k items in the group recommendation list
k = 10
satisfaction = calculate_satisfaction(groupList, users, k)

alfa = max(list(satisfaction.values())) - min(list(satisfaction.values()))

print('\n RESULT: hybrid')
print(groupList)
print(satisfaction)
print('alfa ', alfa)



### ROUND 2
groupList2 = calculate_group_recommendation_list_hybrid(recommendations, alfa)

# calculate satisfaction scores, use only top-k items in the group recommendation list
k = 10
satisfaction2 = calculate_satisfaction(groupList2, users, k)

alfa2 = max(list(satisfaction2.values())) - min(list(satisfaction2.values()))

print('\n RESULT: hybrid, round 2')
print(groupList2)
print(satisfaction2)
print('alfa2 ', alfa2)

