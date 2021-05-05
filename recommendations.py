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

groupType = 'all-similar'

PRINT_RESULTS = True

groups = [
[53478,8,2418,5122,19199],
[34547,56,126,480,1139],
[39617,5,98,251,292],
[13990,37,174,823,1639],
[16341,23,44,251,2548],
[56187,24,314,2397,5916],
[42490,9,44,254,6421],
[33507,8,79,1309,1667],
[61040,3,1524,4289,8082],
[63429,9,44,292,1077],
[71009,6,16,954,2469],
[17397,2,305,568,38173],
[51757,18,370,638,3607],
[61722,8,1667,19199,65729],
[7316,2,77,476,762],
[4244,13,53,400,3062],
[44783,2,126,204,232],
[20444,60,305,332,5890],
[59292,13,206,249,314],
[2295,4,116,4076,4144],
[55305,5,10,107,4622],
[42431,6,53,78,272],
[36225,4,298,507,6592],
[37332,2,84,229,307],
[33245,5,249,314,891],
[266,5,13,140,481],
[16001,56,212,292,5418],
[15138,16,88,801,3565],
[633,3,126,303,2018],
[13679,5,100,107,956],
[64216,18,95,206,232],
[12308,2,19,462,936],
[24574,34,124,1639,6734],
[21307,41,50,160,1797],
[3433,10,345,2376,8459],
[4075,18,272,1301,3114],
[49617,13,151,355,3919],
[21934,18,75,170,254],
[54247,6,78,272,503],
[57324,5,56,100,229],
[22486,6,53,480,834],
[40430,19,393,670,2875],
[60773,18,71,79,95],
[31483,16,160,215,900],
[38309,13,206,210,1041],
[42387,44,79,235,2010],
[56567,50,59,76,160],
[25088,8,238,1218,3153],
[32241,58,255,384,1799],
[14513,13,53,226,314],
[36616,35,192,205,910],
[4821,8,23,917,1867],
[40377,5,22,56,251],
[4723,23,33,716,724],
[46975,2,118,232,547],
[4454,27,173,253,1939],
[54863,9,292,544,2073],
[34176,4,54,1185,2398],
[59257,19,95,294,1607],
[36711,36,160,480,3719],
[19704,41,160,1274,2286],
[71465,22,92,136,2666],
[21212,23,33,126,269],
[10106,38,6847,27038,33606],
[24613,6,160,215,341],
[67925,9,35,389,579],
[53927,6,53,272,296],
[63122,4,267,712,1113],
[3610,5,95,232,891],
[36945,19,354,2327,34834],
[8723,27,77,253,997],
[28097,23,119,145,1286],
[1277,18,100,367,3917],
[52426,7,160,272,275],
[3594,2,84,677,1383],
[2474,60,381,1326,2980],
[37181,11,16,160,571],
[39724,212,292,2596,6508],
[46473,5,10,1023,2689],
[17432,51,1277,3723,5164],
[52557,3,1617,16489,24136],
[23276,50,59,162,321],
[39393,5,13,140,266],
[22125,9,254,389,2091],
[948,12,458,1818,7527],
[20502,19,302,2230,6027],
[29613,13,206,950,3172],
[31551,5,95,296,866],
[51655,18,254,303,1018],
[10650,36,303,480,3956],
[11927,26,59,137,141],
[12874,24,370,617,4513],
[10996,34,12837,17116,20336],
[56063,179,1281,9654,11471],
[40513,4,176,638,2493],
[46323,6,16,954,3246],
[32130,6,162,966,1271],
[2875,16,88,801,3824],
[37132,4,138,338,1382],
[18321,7,509,769,6096]
]

NUMBER_OF_USERS = len(groups[0])
NUMBER_OF_GROUPS = len(groups)
CORRELATION_THRESHOLD = 0.7
MOVIES_IN_COMMON_MINIMUM = 6
RECOMMENDATION_ROUNDS = 15

METHOD_NAMES = [
    'AVERAGE-MIN-DISAGREEMENT',
    'ADJUSTED AVERAGE',
    'HYBRID']
METHODS = { 
    'AVERAGE-MIN-DISAGREEMENT' : calculations.calculate_group_recommendation_list_average_min_disagreement,
    'ADJUSTED AVERAGE' : calculations.calculate_group_recommendation_list_adjusted_average,
    'HYBRID' : calculations.calculate_group_recommendation_list_hybrid}

CALCULATION_FACTORS = {
    'AVERAGE-MIN-DISAGREEMENT' : 200, # k = 200
    'ADJUSTED AVERAGE' : 0.25 # weight = 0.25
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

            ###
            print(f'\nResult for method = {m}')
            print(filteredGroupList)
            ###

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


if (PRINT_RESULTS == True):
    ### SAVE RESULTS TO FILE
    filename = f'results/results_20-03-2021_{groupType}'
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
