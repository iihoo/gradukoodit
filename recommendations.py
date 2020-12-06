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

groups = [
[5, 10, 1154, 6579, 9133],
[22, 85, 1450, 4475, 7278],
[57, 196, 251, 1056, 9685],
[5, 675, 1207, 2597, 12856],
[16, 127, 2715, 3012, 7959],
[7, 32, 7176, 8389, 12075],
[21, 91, 3001, 6562, 9840],
[3, 326, 373, 9387, 11955]
]

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
df_results_sat = pd.DataFrame(index=index, columns=['GroupSatO:HYBRID', 'GroupSatO:MOD-V1', 'GroupSatO:MOD-V2', 'GroupSatO:MOD-V3', 'GroupSatO:MOD-V4'])
df_results_dis = pd.DataFrame(index=index, columns=['GroupDisO:HYBRID', 'GroupDisO:MOD-V1', 'GroupDisO:MOD-V2', 'GroupDisO:MOD-V3', 'GroupDisO:MOD-V4'])
df_results_F_score = pd.DataFrame(index=index, columns=['F-score:HYBRID', 'F-score:MOD-V1', 'F-score:MOD-V2', 'F-score:MOD-V3', 'F-score:MOD-V4'])

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
    satisfactionMod_V1 = {u:1 for u in users}
    satisfactionMod_V2 = {u:1 for u in users}
    satisfactionMod_V3 = {u:1 for u in users}
    satisfactionMod_V4 = {u:1 for u in users}

    # create a DataFrame for satisfaction & dissatisfaction scores
    df_scores_sat = pd.DataFrame(columns=['GroupSat:HYBRID', 'GroupSat:MOD-V1', 'GroupSat:MOD-V2', 'GroupSat:MOD-V3', 'GroupSat:MOD-V4'])
    df_scores_dis = pd.DataFrame(columns=['GroupDis:HYBRID', 'GroupDis:MOD-V1', 'GroupDis:MOD-V2', 'GroupDis:MOD-V3', 'GroupDis:MOD-V4'])

    # initialize list of removed movies for both methods
    removedMoviesHybrid = []
    removedMoviesMod_V1 = []
    removedMoviesMod_V2 = []
    removedMoviesMod_V3 = []
    removedMoviesMod_V4 = []

    # simulate recommendation rounds
    for r in range(1, RECOMMENDATION_ROUNDS + 1):
        # calculate group recommendation list(s)
        groupListHybrid = calculations.calculate_group_recommendation_list_hybrid(recommendations, alfa)
        groupListMod_V1 = calculations.calculate_group_recommendation_list_V1(recommendations, satisfactionMod_V1)
        groupListMod_V2 = calculations.calculate_group_recommendation_list_V2(recommendations, satisfactionMod_V2)
        groupListMod_V3 = calculations.calculate_group_recommendation_list_V3(recommendations, satisfactionMod_V3)
        groupListMod_V4 = calculations.calculate_group_recommendation_list_V4(recommendations, satisfactionMod_V4)

        # filter movies that have already been recommended in the previous rounds
        groupListHybridResult = groupListHybrid[~groupListHybrid.movieId.isin(removedMoviesHybrid)]
        groupListMod_V1Result = groupListMod_V1[~groupListMod_V1.movieId.isin(removedMoviesMod_V1)]
        groupListMod_V2Result = groupListMod_V2[~groupListMod_V2.movieId.isin(removedMoviesMod_V2)]
        groupListMod_V3Result = groupListMod_V3[~groupListMod_V3.movieId.isin(removedMoviesMod_V3)]
        groupListMod_V4Result = groupListMod_V4[~groupListMod_V4.movieId.isin(removedMoviesMod_V4)]

        # calculate satisfaction scores, use only top-k items in the group recommendation list
        satisfactionHybrid = calculations.calculate_satisfaction(groupListHybridResult, recommendations, k)
        satisfactionMod_V1 = calculations.calculate_satisfaction(groupListMod_V1Result, recommendations, k)
        satisfactionMod_V2 = calculations.calculate_satisfaction(groupListMod_V2Result, recommendations, k)
        satisfactionMod_V3 = calculations.calculate_satisfaction(groupListMod_V3Result, recommendations, k)
        satisfactionMod_V4 = calculations.calculate_satisfaction(groupListMod_V4Result, recommendations, k)

        # modify alfa value (used in the hybrid method)
        alfa = max(list(satisfactionHybrid.values())) - min(list(satisfactionHybrid.values()))

        # calculate the average satisfaction scores from this round
        groupSatHybrid = sum(satisfactionHybrid.values()) / len(satisfactionHybrid)
        groupSatMod_V1 = sum(satisfactionMod_V1.values()) / len(satisfactionMod_V1)
        groupSatMod_V2 = sum(satisfactionMod_V2.values()) / len(satisfactionMod_V2)
        groupSatMod_V3 = sum(satisfactionMod_V3.values()) / len(satisfactionMod_V3)
        groupSatMod_V4 = sum(satisfactionMod_V4.values()) / len(satisfactionMod_V4)

        # calculate the dissatisfaction scores from this round
        groupDisHybrid = max(satisfactionHybrid.values()) - min(satisfactionHybrid.values())
        groupDisMod_V1 = max(satisfactionMod_V1.values()) - min(satisfactionMod_V1.values())
        groupDisMod_V2 = max(satisfactionMod_V2.values()) - min(satisfactionMod_V2.values())
        groupDisMod_V3 = max(satisfactionMod_V3.values()) - min(satisfactionMod_V3.values())
        groupDisMod_V4 = max(satisfactionMod_V4.values()) - min(satisfactionMod_V4.values())

        # add to results dataframe
        df_scores_sat.loc[r] = [groupSatHybrid, groupSatMod_V1, groupSatMod_V2, groupSatMod_V3, groupSatMod_V4]
        df_scores_dis.loc[r] = [groupDisHybrid, groupDisMod_V1, groupDisMod_V2, groupDisMod_V3, groupDisMod_V4]

        # add top-k movies as to-be-removed, so as to not recommend same movies in the next round
        removedMoviesHybrid.extend(groupListHybridResult['movieId'][:k].values)
        removedMoviesMod_V1.extend(groupListMod_V1Result['movieId'][:k].values)
        removedMoviesMod_V2.extend(groupListMod_V2Result['movieId'][:k].values)
        removedMoviesMod_V3.extend(groupListMod_V3Result['movieId'][:k].values)
        removedMoviesMod_V4.extend(groupListMod_V4Result['movieId'][:k].values)
        

        # calculate results after 5, 10 and 15 rounds
        if r in [5,10,15]:
            # calculate average of the average of group satisfaction scores
            groupSatOHybrid = round(df_scores_sat['GroupSat:HYBRID'].mean(), 3)
            groupSatOMod_V1 = round(df_scores_sat['GroupSat:MOD-V1'].mean(), 3)
            groupSatOMod_V2 = round(df_scores_sat['GroupSat:MOD-V2'].mean(), 3)
            groupSatOMod_V3 = round(df_scores_sat['GroupSat:MOD-V3'].mean(), 3)
            groupSatOMod_V4 = round(df_scores_sat['GroupSat:MOD-V4'].mean(), 3)

            # calculate average of the group dissatisfaction scores
            groupDisOHybrid = round(df_scores_dis['GroupDis:HYBRID'].mean(), 3)
            groupDisOMod_V1 = round(df_scores_dis['GroupDis:MOD-V1'].mean(), 3)
            groupDisOMod_V2 = round(df_scores_dis['GroupDis:MOD-V2'].mean(), 3)
            groupDisOMod_V3 = round(df_scores_dis['GroupDis:MOD-V3'].mean(), 3)
            groupDisOMod_V4 = round(df_scores_dis['GroupDis:MOD-V4'].mean(), 3)

            # calculate F-scores
            F_hybrid = round(calculations.calculate_F_score(groupSatOHybrid, groupDisOHybrid), 3)
            F_Mod_V1 = round(calculations.calculate_F_score(groupSatOMod_V1, groupDisOMod_V1), 3)
            F_Mod_V2 = round(calculations.calculate_F_score(groupSatOMod_V2, groupDisOMod_V2), 3)
            F_Mod_V3 = round(calculations.calculate_F_score(groupSatOMod_V3, groupDisOMod_V3), 3)
            F_Mod_V4 = round(calculations.calculate_F_score(groupSatOMod_V4, groupDisOMod_V4), 3)

            # add to results dataframe
            df_results_sat.loc[(i + 1,r),:] = [groupSatOHybrid, groupSatOMod_V1, groupSatOMod_V2, groupSatOMod_V3, groupSatOMod_V4]
            df_results_dis.loc[(i + 1,r),:] = [groupDisOHybrid, groupDisOMod_V1, groupDisOMod_V2, groupDisOMod_V3, groupDisOMod_V4]
            df_results_F_score.loc[(i + 1,r),:] = [F_hybrid, F_Mod_V1, F_Mod_V2, F_Mod_V3, F_Mod_V4]

    end = time.time()
    print(f'..calculations finished in {(end - start):.1f} seconds')
    print(f'SIMILARITY MATRIX FOR USERS IN THE GROUP:')
    print(df_group_similarity)

print(f'\nRESULTS FOR {NUMBER_OF_GROUPS} ITERATION ROUNDS')
print(df_results_sat)

print(df_results_dis)

print(f'\nF-Scores:')
print(df_results_F_score)

## PLOT RESULTS (GroupSatO and GroupDisO)
# NOTE plot results for groups 1-9
#visualization.plot_results(df_results)