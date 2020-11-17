import pandas as pd
import numpy as np
import random
import time

from sklearn.preprocessing import MinMaxScaler

import calculations
import similarities
import visualization

# just for wide printouts
pd.set_option('display.expand_frame_repr', False)

NUMBER_OF_USERS = 5
CORRELATION_THRESHOLD = 0.7
MOVIES_IN_COMMON_MINIMUM = 6
RECOMMENDATION_ROUNDS = 5
INITIAL_DATA_CHUNK_SIZE = 100000

start = time.time()

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

# calculate similarities between group users
df_group_similarity = similarities.calculate_group_similarity_matrix(users, df_ratings_initial_chunk)

# calculate individial recommendation lists (a dict, where userId is the key and the recommendation list for that user is the dict value)
recommendations = calculations.calculate_recommendations_all(df_ratings_initial_chunk, scaler, users, MOVIES_IN_COMMON_MINIMUM, CORRELATION_THRESHOLD)

# top-k movies
k = 10

# alfa variable for hybrid aggregation method
alfa = 0

# for the first round, initialize satisfaction score = 1, for each user (modified average aggregation)
satisfactionModifiedAggregation = {u:1 for u in users}

# create a DataFrame for satisfaction & dissatisfaction scores
df_scores = pd.DataFrame(columns=['GroupSat:HYBRID', 'GroupSat:MODIF.AGGR.', 'GroupDis:HYBRID', 'GroupDis:MODIF.AGGR.'])

# simulate recommendation rounds
for i in range(1, RECOMMENDATION_ROUNDS + 1):
    print('\n**********************************************************************************************************************************')
    print(f'ROUND {i}')

    # calculate group recommendation list(s)
    groupListHybrid = calculations.calculate_group_recommendation_list_hybrid(recommendations, alfa)
    groupListModifiedAggregation = calculations.calculate_group_recommendation_list_modified_average_aggregation(recommendations, satisfactionModifiedAggregation)

    # calculate satisfaction scores, use only top-k items in the group recommendation list
    satisfactionHybrid = calculations.calculate_satisfaction(groupListHybrid, recommendations, k)
    satisfactionModifiedAggregation = calculations.calculate_satisfaction(groupListModifiedAggregation, recommendations, k)

    # modify alfa value (used in the hybrid method)
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
    recommendations = calculations.remove_movies(recommendations, [groupListHybrid, groupListModifiedAggregation], k)
    
# create a DataFrame for GroupSatO and GroupDisO results
df_results = pd.DataFrame(columns=['GroupSatO:HYBRID', 'GroupSatO:MODIF.AGGR.', 'GroupDisO:HYBRID', 'GroupDisO:MODIF.AGGR.'])

# calculate average of the average of group satisfaction scores
groupSatOHybrid = df_scores['GroupSat:HYBRID'].mean()
groupSatOModifiedAggregation = df_scores['GroupSat:MODIF.AGGR.'].mean()

# calculate average of the group dissatisfaction scores
groupDisOHybrid = df_scores['GroupDis:HYBRID'].mean()
groupDisOModifiedAggregation = df_scores['GroupDis:MODIF.AGGR.'].mean()

# add to results dataframe
df_results.loc[1] = [groupSatOHybrid, groupSatOModifiedAggregation, groupDisOHybrid, groupDisOModifiedAggregation]

print(f'\nAFTER {RECOMMENDATION_ROUNDS} RECOMMENDATION ROUNDS')

print('\nResults:')
print(df_scores)

print(df_results)

print(f'\nF-score, HYBRID: {calculations.calculate_F_score(groupSatOHybrid, groupDisOHybrid):4f}')
print(f'F-score, MODIFIED AGGREGATION: {calculations.calculate_F_score(groupSatOModifiedAggregation, groupDisOModifiedAggregation):4f}')

print('\nSIMILARITY MATRIX FOR TARGER USERS:')
print(df_group_similarity)

end = time.time()
print(f'\nRecommendations finished in {(end - start):1f} seconds')

### PLOT RESULTS
visualization.plot_satisfaction_dissatisfaction(df_scores)
