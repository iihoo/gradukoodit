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
[2, 2057, 3265, 6981, 12079],
[4336,74,358,4477,9472]
]

NUMBER_OF_USERS = len(groups[0])
NUMBER_OF_GROUPS = len(groups)
CORRELATION_THRESHOLD = 0.7
MOVIES_IN_COMMON_MINIMUM = 6
RECOMMENDATION_ROUNDS = 15
INITIAL_DATA_CHUNK_SIZE = 2000000

METHOD_NAMES = [
    'HYBRID', 
    'AVERAGE-MIN-DISAGREEMENT', 
    'ADJUSTED AVERAGE']
METHODS = { 
    METHOD_NAMES[0] : calculations.calculate_group_recommendation_list_hybrid, 
    METHOD_NAMES[1] : calculations.calculate_group_recommendation_list_average_min_disagreement,
    METHOD_NAMES[2] : calculations.calculate_group_recommendation_list_adjusted_average}

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
columnsGroupSatO = ['GroupSatO:' + method for method in METHOD_NAMES]
columnsGroupDisO = ['GroupDisO:' + method for method in METHOD_NAMES]
columnsFScore = ['F-score:' + method for method in METHOD_NAMES]
df_results_sat = pd.DataFrame(index=index, columns=columnsGroupSatO)
df_results_dis = pd.DataFrame(index=index, columns=columnsGroupDisO)
df_results_F_score = pd.DataFrame(index=index, columns=columnsFScore)

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
    satisfactionScores = {}
    for m in METHOD_NAMES:
        satisfactionScores[m] = {u:1 for u in users}

    # create a DataFrame for satisfaction & dissatisfaction scores
    columnsGroupSat = ['GroupSat:' + method for method in METHOD_NAMES]
    columnsGroupDis = ['GroupDis:' + method for method in METHOD_NAMES]
    df_scores_sat = pd.DataFrame(columns=columnsGroupSat)
    df_scores_dis = pd.DataFrame(columns=columnsGroupDis)

    # initialize list of removed movies for all methods
    moviesAlreadyRecommended = {}
    for m in METHOD_NAMES:
        moviesAlreadyRecommended[m] = []

    groupLists = {}
    groupListResults = {}
    groupSat = {}
    groupSatO = {}
    groupDis = {}
    groupDisO = {}
    F_score = {}
    for m in METHOD_NAMES:
        groupLists[m] = []
        groupListResults[m] = []
        groupSat[m] = 1
        groupSatO[m] = 1
        groupDis[m] = 0
        groupDisO[m] = 0
        F_score[m] = 0

    # simulate recommendation rounds
    for r in range(1, RECOMMENDATION_ROUNDS + 1):
        # calculate group recommendation list(s)
        
        for m in METHOD_NAMES:
            groupRecommendationFunction = METHODS[m]
            if groupRecommendationFunction == calculations.calculate_group_recommendation_list_hybrid:
                groupLists[m] = groupRecommendationFunction(recommendations, alfa, satisfactionScores[m])
            else:
                groupLists[m] = groupRecommendationFunction(recommendations, satisfactionScores[m])

        # filter movies that have already been recommended in the previous rounds
        for m in METHOD_NAMES:
            groupList = groupLists[m]
            filteredGroupList = groupList[~groupList.movieId.isin(moviesAlreadyRecommended[m])]
            groupListResults[m] = filteredGroupList

        # calculate satisfaction scores, use only top-k items in the group recommendation list
        for m in METHOD_NAMES:
            satisfactionScores[m] = calculations.calculate_satisfaction(groupListResults[m], recommendations, k)

        # modify alfa value (used in the hybrid method)
        for m in METHOD_NAMES:
            groupRecommendationFunction = METHODS[m]
            if groupRecommendationFunction == calculations.calculate_group_recommendation_list_hybrid:
                alfa = max(list(satisfactionScores[m].values())) - min(list(satisfactionScores[m].values()))
        #alfa = max(list(satisfactionHybrid.values())) - min(list(satisfactionHybrid.values()))

        # calculate the average satisfaction scores from this round
        # and calculate the dissatisfaction scores from this round
        for m in METHOD_NAMES:
            satisfaction = satisfactionScores[m]
            groupSat[m] = sum(satisfaction.values()) / len(satisfaction)
            groupDis[m] = max(satisfaction.values()) - min(satisfaction.values())

        # add to results dataframe
        groupSatResult = [groupSat[m] for m in METHOD_NAMES]
        groupDisResult = [groupDis[m] for m in METHOD_NAMES]
        df_scores_sat.loc[r] = groupSatResult
        df_scores_dis.loc[r] = groupDisResult

        # add top-k movies as to-be-removed, so as to not recommend same movies in the next round
        for m in METHOD_NAMES:
            moviesAlreadyRecommended[m].extend(groupListResults[m]['movieId'][:k].values)

        # calculate results after 5, 10 and 15 rounds
        if r in [5,10,15]:
            # calculate average of the average of group satisfaction scores
            for m in METHOD_NAMES:
                columnName = 'GroupSat:' + m
                groupSatO[m] = round(df_scores_sat[columnName].mean(), 3)

            # calculate average of the group dissatisfaction scores
            for m in METHOD_NAMES:
                columnName = 'GroupDis:' + m
                groupDisO[m] = round(df_scores_dis[columnName].mean(), 3)

            # calculate F-scores
            for m in METHOD_NAMES:
                F_score[m] = round(calculations.calculate_F_score(groupSatO[m], groupDisO[m]), 3)

            # add to results dataframe
            df_results_sat.loc[(i + 1,r),:] = [groupSatO[m] for m in METHOD_NAMES]

            df_results_dis.loc[(i + 1,r),:] = [groupDisO[m] for m in METHOD_NAMES]
            
            df_results_F_score.loc[(i + 1,r),:] = [F_score[m] for m in METHOD_NAMES]

    end = time.time()
    print(f'..calculations finished in {(end - start):.1f} seconds')
    print(f'SIMILARITY MATRIX FOR USERS IN THE GROUP:')
    print(df_group_similarity)

print(f'\nRESULTS FOR {NUMBER_OF_GROUPS} ITERATION ROUNDS')
print('GROUP SATISFACTION OVERALL')
print(df_results_sat)
print('\n...results grouped per round')
df_results_sat.groupby(level='round').apply(print)
print('\n...and average results per round')
print(df_results_sat.apply(pd.to_numeric).groupby(level='round').agg('mean'))

print('\nGROUP DISAGREEMENT OVERALL')
print(df_results_dis)
print('\n...results grouped per round')
df_results_dis.groupby(level='round').apply(print)
print('\n...and average results per round')
print(df_results_dis.apply(pd.to_numeric).groupby(level='round').agg('mean'))

print(f'\nF-Scores:')
print(df_results_F_score)
print('\n...results grouped per round')
df_results_F_score.groupby(level='round').apply(print)
print('\n...and average results per round')
print(df_results_F_score.apply(pd.to_numeric).groupby(level='round').agg('mean'))

'''
### SAVE RESULTS TO FILE
filename = 'results/results_3+2'
with open(f'{filename}.csv', 'a') as file:
    print(f'number of users in a group: {NUMBER_OF_USERS}, number of groups: {NUMBER_OF_GROUPS}, correlation threshold: {CORRELATION_THRESHOLD}, movies in common minimum: {MOVIES_IN_COMMON_MINIMUM}, recommendation rounds: {RECOMMENDATION_ROUNDS}, ratings data size: {INITIAL_DATA_CHUNK_SIZE}', file=file)

    df_results_sat.to_csv(f'{filename}.csv', mode='a')
    df_results_sat.apply(pd.to_numeric).groupby(level='round').agg('mean').to_csv(f'{filename}.csv', mode='a')

    df_results_dis.to_csv(f'{filename}.csv', mode='a')
    df_results_dis.apply(pd.to_numeric).groupby(level='round').agg('mean').to_csv(f'{filename}.csv', mode='a')

    df_results_F_score.to_csv(f'{filename}.csv', mode='a')    
    df_results_F_score.apply(pd.to_numeric).groupby(level='round').agg('mean').to_csv(f'{filename}.csv', mode='a')
'''



## PLOT RESULTS (GroupSatO and GroupDisO)
# NOTE plot results for groups 1-9
#visualization.plot_results(df_results)
