import pandas as pd
import time

import groupformation
import similarities

## choose which group type to find from data
GROUP_TYPE = 'similar'

# get data
df_ratings = pd.read_table('movielens-10m/ratings.dat', sep='::', usecols=[0,1,2], names=['userId', 'movieId', 'rating'], engine='python')

# main parameters
RATINGS_DATA_SIZE = df_ratings.shape[0]
MOVIES_IN_COMMON_MINIMUM = 6
SIMILARITY_THRESHOLD = 0.5
DISSIMILARITY_THRESHOLD = -0.5

methods = {
    '3+2' : groupformation.create_group_type_3_2,
    '4+1' : groupformation.create_group_type_4_1,
    '3+1+1': groupformation.create_group_type_3_1_1,
    'dissimilar': groupformation.create_group_type_all_dissimilar,
    'similar' : groupformation.create_group_type_all_similar}

print(f'FINDING GROUPS OF GROUP TYPE: {GROUP_TYPE} ....\n')

# create file if it does not exist, otherwise append to file
with open(f'grouptypes-10m/groups_{GROUP_TYPE}.csv', 'a') as file:
    # write parameter info to file
    print(f'Parameters: data size = {RATINGS_DATA_SIZE} ratings, movies in common minimum = {MOVIES_IN_COMMON_MINIMUM}', file=file)

    # for group types 3+2, 4+1 and all-similar
    if GROUP_TYPE == '3+2' or GROUP_TYPE == '4+1' or GROUP_TYPE == 'similar':
        # create groups and append to file
        for i in range(1, 100):
            start = time.time()
            # use the correct method for the group type
            group = methods[GROUP_TYPE](df_ratings, MOVIES_IN_COMMON_MINIMUM, SIMILARITY_THRESHOLD, DISSIMILARITY_THRESHOLD)
            if (group == None):
                print(f'\nNo suitable users could be found at round {i + 1}.')
            else:
                print(similarities.calculate_group_similarity_matrix(group, df_ratings))
                groupToString = ','.join([str(user) for user in group])
                print(groupToString, file=file)
            end = time.time()
            print(f'...took {round(end - start)} seconds.\n')
    # for group types 3+1+1 and all-dissimilar        
    else:
        # create groups and append to file
        for i in range(1, 300):
            start = time.time()
            # use the correct method for the group type
            group = methods[GROUP_TYPE](df_ratings, MOVIES_IN_COMMON_MINIMUM, SIMILARITY_THRESHOLD, DISSIMILARITY_THRESHOLD, file)
            end = time.time()
            print(f'...took {round(end - start)} seconds.\n')    