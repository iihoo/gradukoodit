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

groupType = 'all-dissimilar'

groups = [
[42772, 1451, 34309, 34670, 58136]
]

NUMBER_OF_USERS = len(groups[0])
NUMBER_OF_GROUPS = len(groups)
CORRELATION_THRESHOLD = 0.7
MOVIES_IN_COMMON_MINIMUM = 6
RECOMMENDATION_ROUNDS = 15

'''
METHOD_NAMES = [
    'HYBRID', 
    'AVERAGE-MIN-DISAGREEMENT', 
    'ADJUSTED AVERAGE']
METHODS = { 
    METHOD_NAMES[0] : calculations.calculate_group_recommendation_list_hybrid, 
    METHOD_NAMES[1] : calculations.calculate_group_recommendation_list_average_min_disagreement,
    METHOD_NAMES[2] : calculations.calculate_group_recommendation_list_adjusted_average}
'''

METHOD_NAMES = [
    'AVERAGE-MIN-DISAGREEMENT', # top 200
    'AVERAGE-MIN-DISAGREEMENT VERSION 2', # top 100
    'AVERAGE-MIN-DISAGREEMENT VERSION 3', # top 50
    'AVERAGE-MIN-DISAGREEMENT VERSION 4', # top 300
    'ADJUSTED AVERAGE', # w = 1 + factor
    'ADJUSTED AVERAGE VERSION 2', # w = 1 + 0.5 * factor
    'ADJUSTED AVERAGE VERSION 3', # w = 1 + 0.25 * factor,
    'ADJUSTED AVERAGE VERSION 4', # W = 1 + 1.25 * factor,
    'ADJUSTED AVERAGE VERSION 5', # W = 1 + 1.5 * factor,
    'ADJUSTED AVERAGE VERSION 6', # W = 1 + 2 * factor
    'HYBRID']
METHODS = { 
    METHOD_NAMES[0] : calculations.calculate_group_recommendation_list_average_min_disagreement, 
    METHOD_NAMES[1] : calculations.calculate_group_recommendation_list_average_min_disagreement,
    METHOD_NAMES[2] : calculations.calculate_group_recommendation_list_average_min_disagreement,
    METHOD_NAMES[3] : calculations.calculate_group_recommendation_list_average_min_disagreement,
    METHOD_NAMES[4] : calculations.calculate_group_recommendation_list_adjusted_average,
    METHOD_NAMES[5] : calculations.calculate_group_recommendation_list_adjusted_average,
    METHOD_NAMES[6] : calculations.calculate_group_recommendation_list_adjusted_average,
    METHOD_NAMES[7] : calculations.calculate_group_recommendation_list_adjusted_average,
    METHOD_NAMES[8] : calculations.calculate_group_recommendation_list_adjusted_average,
    METHOD_NAMES[9] : calculations.calculate_group_recommendation_list_adjusted_average,
    METHOD_NAMES[10] : calculations.calculate_group_recommendation_list_hybrid}

CALCULATION_FACTORS = {
    METHOD_NAMES[0] : 200,
    METHOD_NAMES[1] : 100,
    METHOD_NAMES[2] : 50,
    METHOD_NAMES[3] : 300,
    METHOD_NAMES[4] : 0,
    METHOD_NAMES[5] : 0.5,
    METHOD_NAMES[6] : 0.25,
    METHOD_NAMES[7] : 1.25,
    METHOD_NAMES[8] : 1.5,
    METHOD_NAMES[9] : 2.0 
}

df_ratings = pd.read_table('movielens-10m/ratings.dat', sep='::', usecols=[0,1,2], names=['userId', 'movieId', 'rating'], engine='python')

# get scale of ratings
ratingScale = df_ratings['rating'].unique()
ratingScale.sort()
ratingScale = (ratingScale[0], ratingScale[len(ratingScale) - 1])
scaler = MinMaxScaler(feature_range=(ratingScale))

# create a DataFrame for GroupSatO and GroupDisO results
index = pd.MultiIndex.from_tuples([(1,5)], names=('group', 'round'))
columnsGroupSatO = ['GroupSatO:' + method for method in METHOD_NAMES]
df_results_sat = pd.DataFrame(index=index, columns=columnsGroupSatO)

columnsGroupDisO = ['GroupDisO:' + method for method in METHOD_NAMES]
df_results_dis = pd.DataFrame(index=index, columns=columnsGroupDisO)

columnsNDCGO = ['NDCG:' + method for method in METHOD_NAMES]
df_results_ndcg = pd.DataFrame(index=index, columns=columnsNDCGO)

columnsFScore = ['F-score:' + method for method in METHOD_NAMES]
df_results_F_score = pd.DataFrame(index=index, columns=columnsFScore)

# calculate recommendations for each group, one by one
for i in range(0, NUMBER_OF_GROUPS):
    # pick one group for this iteration
    users = groups[i]

    # calculate similarity matrix for the group (just for printing)
    df_group_similarity = similarities.calculate_group_similarity_matrix(users, df_ratings)

    print(f'\nCalculating recommendations for group {i + 1}...')
    start = time.time()

    # calculate individial recommendation lists (a dict, where userId is the key and the recommendation list for that user is the dict value)
    recommendations = calculations.calculate_recommendations_all(df_ratings, scaler, users, MOVIES_IN_COMMON_MINIMUM, CORRELATION_THRESHOLD)

    # top-k movies
    k = 10

    # alfa variable for hybrid aggregation method
    alfa = 0

    # create a DataFrame for satisfaction & dissatisfaction scores
    columnsGroupSat = ['GroupSat:' + method for method in METHOD_NAMES]
    df_scores_sat = pd.DataFrame(columns=columnsGroupSat)

    columnsGroupDis = ['GroupDis:' + method for method in METHOD_NAMES]
    df_scores_dis = pd.DataFrame(columns=columnsGroupDis)

    columnsNDCG = ['NDCG:' + method for method in METHOD_NAMES]
    df_scores_ndcg = pd.DataFrame(columns=columnsNDCG)

    # initialize parameters 
    moviesAlreadyRecommended = {}        
    satisfactionScores = {}
    ndcgScores = {}
    groupLists = {}
    groupListResults = {}
    groupSat = {}
    groupSatO = {}
    groupDis = {}
    groupDisO = {}
    groupNDCG = {}
    groupNDCGO = {}
    F_score = {}

    for m in METHOD_NAMES:
        moviesAlreadyRecommended[m] = []
        satisfactionScores[m] = {u:1 for u in users}
        ndcgScores[m] = {u:1 for u in users}
        groupLists[m] = []
        groupListResults[m] = []
        groupSat[m] = 1
        groupSatO[m] = 1
        groupDis[m] = 0
        groupDisO[m] = 0
        groupNDCG[m] = 1
        groupNDCGO[m] = 1
        F_score[m] = 0

    # calculate recommendation rounds, one by one, and save results
    for r in range(1, RECOMMENDATION_ROUNDS + 1):
        # calculate recommendation results in this round for each method, one by one
        for m in METHOD_NAMES:
            # calculate group recommendation list(s)
            groupRecommendationFunction = METHODS[m]
            if groupRecommendationFunction == calculations.calculate_group_recommendation_list_hybrid:
                groupLists[m] = groupRecommendationFunction(recommendations, alfa, satisfactionScores[m])
            else:
                groupLists[m] = groupRecommendationFunction(recommendations, satisfactionScores[m], CALCULATION_FACTORS[m])

            # filter movies that have already been recommended in the previous rounds
            groupList = groupLists[m]
            filteredGroupList = groupList[~groupList.movieId.isin(moviesAlreadyRecommended[m])]
            groupListResults[m] = filteredGroupList

            # calculate satisfaction scores, use only top-k items in the group recommendation list
            satisfactionScores[m] = calculations.calculate_satisfaction(groupListResults[m], recommendations, k)

            # NDCG scores, use only top-k items in the group recommendation list
            ndcgScores[m] = calculations.calculate_ndcg(groupListResults[m], users, k)

            # modify alfa value (used in the hybrid method)
            if groupRecommendationFunction == calculations.calculate_group_recommendation_list_hybrid:
                alfa = max(list(satisfactionScores[m].values())) - min(list(satisfactionScores[m].values()))

            # calculate the average satisfaction scores from this round
            # and calculate the dissatisfaction scores from this round
            satisfaction = satisfactionScores[m]
            groupSat[m] = sum(satisfaction.values()) / len(satisfaction)
            groupDis[m] = max(satisfaction.values()) - min(satisfaction.values())

            # calculate the average NDCG scores from this round
            ndcg = ndcgScores[m]
            groupNDCG[m] = sum(ndcg.values()) / len(ndcg)

            # add top-k movies to moviesAlreadyRecommended, so as to not recommend same movies in the next round
            moviesAlreadyRecommended[m].extend(groupListResults[m]['movieId'][:k].values)

        # add to results dataframe
        groupSatResult = [groupSat[m] for m in METHOD_NAMES]
        df_scores_sat.loc[r] = groupSatResult

        groupDisResult = [groupDis[m] for m in METHOD_NAMES]
        df_scores_dis.loc[r] = groupDisResult    

        groupNDCGResult = [groupNDCG[m] for m in METHOD_NAMES]
        df_scores_ndcg.loc[r] = groupNDCGResult  

        # calculate results after 5, 10 and 15 rounds
        if r in [5,10,15]:
            
            for m in METHOD_NAMES:
                # calculate average of the average of group satisfaction scores
                columnNameGroupSat = 'GroupSat:' + m
                groupSatO[m] = round(df_scores_sat[columnNameGroupSat].mean(), 3)

                # calculate average of the group dissatisfaction scores
                columnNameGroupDis = 'GroupDis:' + m
                groupDisO[m] = round(df_scores_dis[columnNameGroupDis].mean(), 3)

                columnNameGroupNDCG = 'NDCG:' + m
                groupNDCGO[m] = round(df_scores_ndcg[columnNameGroupNDCG].mean(), 3)

                # calculate F-scores
                F_score[m] = round(calculations.calculate_F_score(groupSatO[m], groupDisO[m]), 3)

            # add to results dataframe
            df_results_sat.loc[(i + 1,r),:] = [groupSatO[m] for m in METHOD_NAMES]
            df_results_dis.loc[(i + 1,r),:] = [groupDisO[m] for m in METHOD_NAMES]
            df_results_ndcg.loc[(i + 1,r),:] = [groupNDCGO[m] for m in METHOD_NAMES]
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

print('\nGROUP NDCG OVERALL')
print(df_results_ndcg)
print('\n...results grouped per round')
df_results_ndcg.groupby(level='round').apply(print)
print('\n...and average results per round')
print(df_results_ndcg.apply(pd.to_numeric).groupby(level='round').agg('mean'))

print(f'\nF-Scores:')
print(df_results_F_score)
print('\n...results grouped per round')
df_results_F_score.groupby(level='round').apply(print)
print('\n...and average results per round')
print(df_results_F_score.apply(pd.to_numeric).groupby(level='round').agg('mean'))


### SAVE RESULTS TO FILE
filename = f'results/results_{groupType}'
with open(f'{filename}.csv', 'a') as file:
    print(f'group type: {groupType}, number of groups: {len(groups)}, number of users in a group: {NUMBER_OF_USERS}, number of groups: {NUMBER_OF_GROUPS}, correlation threshold: {CORRELATION_THRESHOLD}, movies in common minimum: {MOVIES_IN_COMMON_MINIMUM}, recommendation rounds: {RECOMMENDATION_ROUNDS}, ratings data size: {df_ratings.shape[0]}', file=file)

    #df_results_sat.to_csv(f'{filename}.csv', mode='a')
    df_results_sat.apply(pd.to_numeric).groupby(level='round').agg('mean').to_csv(f'{filename}.csv', mode='a')

    #df_results_dis.to_csv(f'{filename}.csv', mode='a')
    df_results_dis.apply(pd.to_numeric).groupby(level='round').agg('mean').to_csv(f'{filename}.csv', mode='a')

    #df_results_ndcg.to_csv(f'{filename}.csv', mode='a')
    df_results_ndcg.apply(pd.to_numeric).groupby(level='round').agg('mean').to_csv(f'{filename}.csv', mode='a')

    #df_results_F_score.to_csv(f'{filename}.csv', mode='a')    
    df_results_F_score.apply(pd.to_numeric).groupby(level='round').agg('mean').to_csv(f'{filename}.csv', mode='a')
