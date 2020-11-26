import pandas as pd
import numpy as np
import random
import time
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

import calculations
import similarities
import visualization

# just for wide printouts
pd.set_option('display.expand_frame_repr', False)

groups = [[16, 127, 2715, 3012, 7959],
[7, 32, 7176, 8389, 12075],
[21, 91, 3001, 6562, 9840],
[3, 326, 373, 9387, 11955],
[20, 424, 4422, 6646, 13138],
[44, 108, 5395, 7818, 11649],
[33, 710, 755, 2678, 4806],
[2, 117, 2095, 5573, 6846],
[164, 571, 2320, 5583, 10787],]

NUMBER_OF_USERS = len(groups[0])
NUMBER_OF_GROUPS = len(groups)
CORRELATION_THRESHOLD = 0.7
MOVIES_IN_COMMON_MINIMUM = 6
RECOMMENDATION_ROUNDS = 15
INITIAL_DATA_CHUNK_SIZE = 2000000

# get initial data chunk 
initialRatingsDataChunk = pd.read_csv('movielens-25m/ratings.csv', usecols=['userId', 'movieId', 'rating'], chunksize=INITIAL_DATA_CHUNK_SIZE)
df_ratings_initial_chunk = initialRatingsDataChunk.get_chunk()

# get scale of ratings
ratingScale = df_ratings_initial_chunk['rating'].unique()
ratingScale.sort()
ratingScale = (ratingScale[0], ratingScale[len(ratingScale) - 1])
scaler = MinMaxScaler(feature_range=(ratingScale))

# create a DataFrame for GroupSatO and GroupDisO results
index = pd.MultiIndex.from_tuples([(1,5)], names=('group', 'round'))
df_results = pd.DataFrame(index=index, columns=['GroupSatO:HYBRID', 'GroupSatO:MODIF.AGGR.', 'GroupDisO:HYBRID', 'GroupDisO:MODIF.AGGR.', 'F-score:HYBRID', 'F-score:MODIF.AGGR.'])

for i in range(0, NUMBER_OF_GROUPS):
    # pick one group for this iteration
    users = groups[i]

    # calculate similarity matrix for the group
    df_group_similarity = similarities.calculate_group_similarity_matrix(users, df_ratings_initial_chunk)

    print(f'\nCalculating recommendations for group {i + 1}...')
    start = time.time()

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

    # initialize list of removed movies for both methods
    removedMoviesHybrid = []
    removedMoviesModifiedAggregation = []

    # simulate recommendation rounds
    for r in range(1, RECOMMENDATION_ROUNDS + 1):
        # calculate group recommendation list(s)
        groupListHybrid = calculations.calculate_group_recommendation_list_hybrid(recommendations, alfa)
        groupListModifiedAggregation = calculations.calculate_group_recommendation_list_modified_average_aggregation(recommendations, satisfactionModifiedAggregation)

        # filter movies that have already been recommended in the previous rounds
        groupListHybridResult = groupListHybrid[~groupListHybrid.movieId.isin(removedMoviesHybrid)]
        groupListModifiedAggregationResult = groupListModifiedAggregation[~groupListModifiedAggregation.movieId.isin(removedMoviesModifiedAggregation)]

        # calculate satisfaction scores, use only top-k items in the group recommendation list
        satisfactionHybrid = calculations.calculate_satisfaction(groupListHybridResult, recommendations, k)
        satisfactionModifiedAggregation = calculations.calculate_satisfaction(groupListModifiedAggregationResult, recommendations, k)

        # modify alfa value (used in the hybrid method)
        alfa = max(list(satisfactionHybrid.values())) - min(list(satisfactionHybrid.values()))

        # calculate the average satisfaction scores from this round
        groupSatHybrid = sum(satisfactionHybrid.values()) / len(satisfactionHybrid)
        groupSatModifiedAggregation = sum(satisfactionModifiedAggregation.values()) / len(satisfactionModifiedAggregation)

        # calculate the dissatisfaction scores from this round
        groupDisHybrid = max(satisfactionHybrid.values()) - min(satisfactionHybrid.values())
        groupDisModifiedAggregation = max(satisfactionModifiedAggregation.values()) - min(satisfactionModifiedAggregation.values())

        # add to results dataframe
        df_scores.loc[r] = [groupSatHybrid, groupSatModifiedAggregation, groupDisHybrid, groupDisModifiedAggregation]

        # add top-k movies as to-be-removed, so as to not recommend same movies in the next round
        removedMoviesHybrid.extend(groupListHybridResult['movieId'][:k].values)
        removedMoviesModifiedAggregation.extend(groupListModifiedAggregationResult['movieId'][:k].values)

        # calculate results after 5, 10 and 15 rounds
        if r in [5,10,15]:
            # calculate average of the average of group satisfaction scores
            groupSatOHybrid = round(df_scores['GroupSat:HYBRID'].mean(), 3)
            groupSatOModifiedAggregation = round(df_scores['GroupSat:MODIF.AGGR.'].mean(), 3)

            # calculate average of the group dissatisfaction scores
            groupDisOHybrid = round(df_scores['GroupDis:HYBRID'].mean(), 3)
            groupDisOModifiedAggregation = round(df_scores['GroupDis:MODIF.AGGR.'].mean(), 3)

            # calculate F-scores
            F_hybrid = round(calculations.calculate_F_score(groupSatOHybrid, groupDisOHybrid), 3)
            F_ModifiedAggregation = round(calculations.calculate_F_score(groupSatOModifiedAggregation, groupDisOModifiedAggregation), 3)

            # add to results dataframe
            df_results.loc[(i + 1,r),:] = [groupSatOHybrid, groupSatOModifiedAggregation, groupDisOHybrid, groupDisOModifiedAggregation, F_hybrid, F_ModifiedAggregation]

    end = time.time()
    print(f'..calculations finished in {(end - start):.1f} seconds')
    print(f'SIMILARITY MATRIX FOR USERS IN THE GROUP:')
    print(df_group_similarity)

print(f'\nRESULTS FOR {NUMBER_OF_GROUPS} ITERATION ROUNDS')
print(df_results)

## PLOT RESULTS (GroupSatO and GroupDisO)
# NOTE plot results for groups 1-9
visualization.plot_results(df_results)