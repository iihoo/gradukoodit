import pandas as pd
import time

import groupformation
import similarities

INITIAL_DATA_CHUNK_SIZE = 2000000
MOVIES_IN_COMMON_MINIMUM = 6
SIMILARITY_THRESHOLD = 0.5
DISSIMILARITY_THRESHOLD = -0.5

# get initial data chunk 
initialRatingsDataChunk = pd.read_csv('movielens-25m/ratings.csv', usecols=['userId', 'movieId', 'rating'], chunksize=INITIAL_DATA_CHUNK_SIZE)
df_ratings_initial_chunk = initialRatingsDataChunk.get_chunk()

# create file if it does not exist, otherwise append to file
with open('grouptypes/groups_dissimilar.csv', 'a') as file:
    # write parameter info to file
    print(f'Parameters: data size = {INITIAL_DATA_CHUNK_SIZE}, movies in common minimum = {MOVIES_IN_COMMON_MINIMUM}', file=file)

    # create groups and append to file
    for i in range(1, 100):
        userId = i
        start = time.time()
        group = groupformation.create_group_type_all_dissimilar(userId, df_ratings_initial_chunk, MOVIES_IN_COMMON_MINIMUM, DISSIMILARITY_THRESHOLD, file)
        end = time.time()
        #if (group == None):
        #    print(f'\nNo suitable users could be found at round {i + 1}.')
        #else:
        #    print(similarities.calculate_group_similarity_matrix(group, df_ratings_initial_chunk))
        #    groupToString = ','.join([str(user) for user in group])
        #    print(groupToString, file=file)
        print(f'...took {round(end - start)} seconds.\n')