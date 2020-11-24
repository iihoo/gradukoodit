import pandas as pd

import groupformation
import similarities

INITIAL_DATA_CHUNK_SIZE = 2000000
MOVIES_IN_COMMON_MINIMUM = 6
SIMILARITY_THRESHOLD = 0.5
DISSIMILARITY_THRESHOLD = -0.5

# get initial data chunk 
initialRatingsDataChunk = pd.read_csv('movielens-25m/ratings.csv', usecols=['userId', 'movieId', 'rating'], chunksize=INITIAL_DATA_CHUNK_SIZE)
df_ratings_initial_chunk = initialRatingsDataChunk.get_chunk()
'''
# create file, return error if exists already
with open('grouptypes/group-3-2.csv', 'x') as file:
    # write parameter info to file
    print(f'Parameters: data size = {INITIAL_DATA_CHUNK_SIZE}, movies in common minimum = {MOVIES_IN_COMMON_MINIMUM}', file=file)

    # create groups and append to file
    for i in range(0, 1):
        group = groupformation.create_group_type_3_2(df_ratings_initial_chunk, MOVIES_IN_COMMON_MINIMUM)
        print(similarities.calculate_group_similarity_matrix(group, df_ratings_initial_chunk))
        groupToString = ','.join([str(user) for user in group])
        print(groupToString, file=file)
'''

for i in range(0, 3):
        group = groupformation.create_group_type_3_2(df_ratings_initial_chunk, MOVIES_IN_COMMON_MINIMUM, SIMILARITY_THRESHOLD, DISSIMILARITY_THRESHOLD)
        if (group == None):
            print(f'\nNo suitable users could be found at round {i + 1}.\n')
        else:
            print(similarities.calculate_group_similarity_matrix(group, df_ratings_initial_chunk))
            groupToString = ','.join([str(user) for user in group])