import pandas as pd
import random
import time

import similarities

# create a group type 3 + 2
def create_group_type_3_2(df_ratings, moviesInCommonMinimum, similarityThreshold, dissimilarityThreshold):
    """
    Create a group with two subgroups, with one subgroup consisting of three users and one subgroup consisting of two users.
    
    Users in a subgroup are similar with each other, but dissimilar to all users in other subgroups.
    
    Users are considered similar when they have a Pearson Correlation value above 'similarityThreshold', and dissimilar when they have a Pearson Correlation value below 'dissimilarityThreshold'.
    """
    group = []

    # pick the first user randomly and add to group
    user1 = random.choice(df_ratings['userId'].unique().tolist())
    group.append(user1)    

    # get similar and dissimilar users for the group member 1: save to separate dataframes
    user = group[0]
    df_1 = similarities.similarity_values(df_ratings, user, moviesInCommonMinimum)
    df_1.rename(columns={'PearsonCorrelation': f'PearsonCorrelation_{user}'}, inplace=True)
    df_similar_1 = df_1[df_1[f'PearsonCorrelation_{user}'] >= similarityThreshold]
    df_dissimilar_1 = df_1[df_1[f'PearsonCorrelation_{user}'] <= dissimilarityThreshold]
    
    # get group member 2 from the similar user for group member 1, stop searching if no similar/dissimilar users are found
    if (df_similar_1.shape[0] < 1):
        return None
    group.append(int(df_similar_1.iloc[0]['userId']))

    # get similar and dissimilar users for the group member 2: save to separate dataframes
    user = group[1]
    df_2 = similarities.similarity_values(df_ratings, user, moviesInCommonMinimum)
    df_2.rename(columns={'PearsonCorrelation': f'PearsonCorrelation_{user}'}, inplace=True)
    df_similar_2 = df_2[df_2[f'PearsonCorrelation_{user}'] >= similarityThreshold]
    df_dissimilar_2 = df_2[df_2[f'PearsonCorrelation_{user}'] <= dissimilarityThreshold]

    # merge similar users for both group member 1 and group member 2
    df_merged_1 = df_similar_1.merge(df_similar_2, left_on='userId', right_on='userId')

    # and get group member 3, stop searching if no similar/dissimilar users are found
    if (df_merged_1.shape[0] < 1):
        return None
    group.append(int(df_merged_1.iloc[0]['userId']))

    # get similar and dissimilar users for the group member 3: save to separate dataframes
    user = group[2]
    df_3 = similarities.similarity_values(df_ratings, user, moviesInCommonMinimum)
    df_3.rename(columns={'PearsonCorrelation': f'PearsonCorrelation_{user}'}, inplace=True)
    df_similar_3 = df_3[df_3[f'PearsonCorrelation_{user}'] >= similarityThreshold]
    df_dissimilar_3 = df_3[df_3[f'PearsonCorrelation_{user}'] <= dissimilarityThreshold]

    # get dissimilar users for group members 1, 2 and 3 by merging the dissimilar user dataframes
    df_merged_2 = df_dissimilar_1.merge(df_dissimilar_2, left_on='userId', right_on='userId')
    df_merged_2 = df_merged_2.merge(df_dissimilar_3, left_on='userId', right_on='userId')
    
    # and get group member 4, stop searching if no similar/dissimilar users are found
    if (df_merged_2.shape[0] < 1):
        return None
    group.append(int(df_merged_2.iloc[0]['userId']))

    # get similar and dissimilar users for the group member 4: save to separate dataframes
    user = group[3]
    df_4 = similarities.similarity_values(df_ratings, user, moviesInCommonMinimum)
    df_4.rename(columns={'PearsonCorrelation': f'PearsonCorrelation_{user}'}, inplace=True)
    df_similar_4 = df_4[df_4[f'PearsonCorrelation_{user}'] >= similarityThreshold]
    df_dissimilar_4 = df_4[df_4[f'PearsonCorrelation_{user}'] <= dissimilarityThreshold]

    # get dissimilar users for group members 1, 2 and 3 and merge with similar user for group member 4
    df_merged_3 = df_merged_2.merge(df_similar_4, left_on='userId', right_on='userId')

    # and get group member 5, stop searching if no similar/dissimilar users are found
    if (df_merged_3.shape[0] < 1):
        return None
    group.append(int(df_merged_3.iloc[0]['userId']))

    return group


# create a group type 4 + 1
def create_group_type_4_1(df_ratings, moviesInCommonMinimum, similarityThreshold, dissimilarityThreshold):
    """
    Create a group with two subgroups, with one subgroup consisting of four users and one one-person subgroup.
    
    Users in a subgroup are similar with each other, but dissimilar to all users in other subgroups.
    
    Users are considered similar when they have a Pearson Correlation value above 'similarityThreshold', and dissimilar when they have a Pearson Correlation value below 'dissimilarityThreshold'.
    """
    group = []

    # pick the first user randomly and add to group
    user1 = random.choice(df_ratings['userId'].unique().tolist())
    group.append(user1)    

    # get similar and dissimilar users for the group member 1: save to separate dataframes
    user = group[0]
    df_1 = similarities.similarity_values(df_ratings, user, moviesInCommonMinimum)
    df_1.rename(columns={'PearsonCorrelation': f'PearsonCorrelation_{user}'}, inplace=True)
    df_similar_1 = df_1[df_1[f'PearsonCorrelation_{user}'] >= similarityThreshold]
    df_dissimilar_1 = df_1[df_1[f'PearsonCorrelation_{user}'] <= dissimilarityThreshold]
    
    # get group member 2 from the similar user for group member 1, stop searching if no similar/dissimilar users are found
    if (df_similar_1.shape[0] < 1):
        return None
    group.append(int(df_similar_1.iloc[0]['userId']))

    # get similar and dissimilar users for the group member 2: save to separate dataframes
    user = group[1]
    df_2 = similarities.similarity_values(df_ratings, user, moviesInCommonMinimum)
    df_2.rename(columns={'PearsonCorrelation': f'PearsonCorrelation_{user}'}, inplace=True)
    df_similar_2 = df_2[df_2[f'PearsonCorrelation_{user}'] >= similarityThreshold]
    df_dissimilar_2 = df_2[df_2[f'PearsonCorrelation_{user}'] <= dissimilarityThreshold]

    # merge similar users for both group member 1 and group member 2
    df_merged_1 = df_similar_1.merge(df_similar_2, left_on='userId', right_on='userId')

    # and get group member 3, stop searching if no similar/dissimilar users are found
    if (df_merged_1.shape[0] < 1):
        return None
    group.append(int(df_merged_1.iloc[0]['userId']))

    # get similar and dissimilar users for the group member 3: save to separate dataframes
    user = group[2]
    df_3 = similarities.similarity_values(df_ratings, user, moviesInCommonMinimum)
    df_3.rename(columns={'PearsonCorrelation': f'PearsonCorrelation_{user}'}, inplace=True)
    df_similar_3 = df_3[df_3[f'PearsonCorrelation_{user}'] >= similarityThreshold]
    df_dissimilar_3 = df_3[df_3[f'PearsonCorrelation_{user}'] <= dissimilarityThreshold]

    # merge similar users for group members 1, 2 and 3
    df_merged_2 = df_merged_1.merge(df_similar_3, left_on='userId', right_on='userId')
    
    # and get group member 4, stop searching if no similar/dissimilar users are found
    if (df_merged_2.shape[0] < 1):
        return None
    group.append(int(df_merged_2.iloc[0]['userId']))

    # get similar and dissimilar users for the group member 4: save to separate dataframes
    user = group[3]
    df_4 = similarities.similarity_values(df_ratings, user, moviesInCommonMinimum)
    df_4.rename(columns={'PearsonCorrelation': f'PearsonCorrelation_{user}'}, inplace=True)
    df_similar_4 = df_4[df_4[f'PearsonCorrelation_{user}'] >= similarityThreshold]
    df_dissimilar_4 = df_4[df_4[f'PearsonCorrelation_{user}'] <= dissimilarityThreshold]

    # get dissimilar users for group members 1, 2, 3 and 4 
    df_merged_3 = df_dissimilar_1.merge(df_dissimilar_2, left_on='userId', right_on='userId')
    df_merged_3 = df_merged_3.merge(df_dissimilar_3, left_on='userId', right_on='userId')
    df_merged_3 = df_merged_3.merge(df_dissimilar_4, left_on='userId', right_on='userId')

    # and get group member 5, stop searching if no similar/dissimilar users are found
    if (df_merged_3.shape[0] < 1):
        return None
    group.append(int(df_merged_3.iloc[0]['userId']))

    return group

# create a group type 3 + 1 + 1
def create_group_type_3_1_1(df_ratings, moviesInCommonMinimum, similarityThreshold, dissimilarityThreshold):
    """
    Create a group with three subgroups, with one subgroup consisting of three users and two one-person subgroups.
    
    Users in a subgroup are similar with each other, but dissimilar to all users in other subgroups.
    
    Users are considered similar when they have a Pearson Correlation value above 'similarityThreshold', and dissimilar when they have a Pearson Correlation value below 'dissimilarityThreshold'.
    """
    group = []

    # pick the first user randomly and add to group
    user1 = random.choice(df_ratings['userId'].unique().tolist())
    group.append(user1)    

    # get similar and dissimilar users for the group member 1: save to separate dataframes
    user = group[0]
    df_1 = similarities.similarity_values(df_ratings, user, moviesInCommonMinimum)
    df_1.rename(columns={'PearsonCorrelation': f'PearsonCorrelation_{user}'}, inplace=True)
    df_similar_1 = df_1[df_1[f'PearsonCorrelation_{user}'] >= similarityThreshold]
    df_dissimilar_1 = df_1[df_1[f'PearsonCorrelation_{user}'] <= dissimilarityThreshold]
    
    # get group member 2 from the similar user for group member 1, stop searching if no similar/dissimilar users are found
    if (df_similar_1.shape[0] < 1):
        return None
    group.append(int(df_similar_1.iloc[0]['userId']))

    # get similar and dissimilar users for the group member 2: save to separate dataframes
    user = group[1]
    df_2 = similarities.similarity_values(df_ratings, user, moviesInCommonMinimum)
    df_2.rename(columns={'PearsonCorrelation': f'PearsonCorrelation_{user}'}, inplace=True)
    df_similar_2 = df_2[df_2[f'PearsonCorrelation_{user}'] >= similarityThreshold]
    df_dissimilar_2 = df_2[df_2[f'PearsonCorrelation_{user}'] <= dissimilarityThreshold]

    # merge similar users for both group member 1 and group member 2
    df_merged_1 = df_similar_1.merge(df_similar_2, left_on='userId', right_on='userId')

    # and get group member 3, stop searching if no similar/dissimilar users are found
    if (df_merged_1.shape[0] < 1):
        return None
    group.append(int(df_merged_1.iloc[0]['userId']))

    # get similar and dissimilar users for the group member 3: save to separate dataframes
    user = group[2]
    df_3 = similarities.similarity_values(df_ratings, user, moviesInCommonMinimum)
    df_3.rename(columns={'PearsonCorrelation': f'PearsonCorrelation_{user}'}, inplace=True)
    df_similar_3 = df_3[df_3[f'PearsonCorrelation_{user}'] >= similarityThreshold]
    df_dissimilar_3 = df_3[df_3[f'PearsonCorrelation_{user}'] <= dissimilarityThreshold]

    # get dissimilar users for group members 1, 2 and 3 by merging the dissimilar user dataframes
    df_merged_2 = df_dissimilar_1.merge(df_dissimilar_2, left_on='userId', right_on='userId')
    df_merged_2 = df_merged_2.merge(df_dissimilar_3, left_on='userId', right_on='userId')
    
    # and get group member 4, stop searching if no similar/dissimilar users are found
    if (df_merged_2.shape[0] < 1):
        return None
    group.append(int(df_merged_2.iloc[0]['userId']))

    # get similar and dissimilar users for the group member 4: save to separate dataframes
    user = group[3]
    df_4 = similarities.similarity_values(df_ratings, user, moviesInCommonMinimum)
    df_4.rename(columns={'PearsonCorrelation': f'PearsonCorrelation_{user}'}, inplace=True)
    df_similar_4 = df_4[df_4[f'PearsonCorrelation_{user}'] >= similarityThreshold]
    df_dissimilar_4 = df_4[df_4[f'PearsonCorrelation_{user}'] <= dissimilarityThreshold]

    # get dissimilar users for group members 1, 2, 3 and 4 by merging the dissimilar user dataframes
    df_merged_3 = df_merged_2.merge(df_dissimilar_4, left_on='userId', right_on='userId')

    # and get group member 5, stop searching if no similar/dissimilar users are found
    if (df_merged_3.shape[0] < 1):
        return None
    group.append(int(df_merged_3.iloc[0]['userId']))

    return group

# create a group type where all users are dissimilar with each other, and write results to file
def create_group_type_all_dissimilar(user1, df_ratings, moviesInCommonMinimum, dissimilarityThreshold, file):
    """
    Create a group where all users are dissimilar with each other.

    Write results to 'file'.
    
    Users are considered dissimilar when they have a Pearson Correlation value below 'dissimilarityThreshold'.
    """

    MOVIES_IN_COMMON_MINIMUM = moviesInCommonMinimum
    DISSIMILARITY_THRESHOLD = dissimilarityThreshold
    DF_RATINGS = df_ratings

    userPool = DF_RATINGS[DF_RATINGS['userId'] != user1]['userId'].unique().tolist()

    def dissimilar(user, userPool):
        df = similarity_values_for_group_formation(user, userPool, DF_RATINGS, MOVIES_IN_COMMON_MINIMUM)
        df.rename(columns={'PearsonCorrelation': user}, inplace=True)
        df_dissimilar = df[df[user] <= DISSIMILARITY_THRESHOLD]
        return df_dissimilar

    def find_all_dissimilar(df_result):
        print(df_result)
        # return False if a group cannot be formed (if from the remaining dissimilar users it is impossible to get 5)
        if len(df_result.columns) + df_result.shape[0] < 6:
            return False
        elif len(df_result.columns) == 5:
            group = df_result.columns[1:].tolist()
            group.append(int(df_result.iloc[0]['userId']))
            print(f'\n Group formed: {group}\n')
            print(group, file=file)
            return True
        else:
            for i in range(0, len(df_result.index)):
                user = int(df_result.iloc[i]['userId'])
                userPool = df_result.iloc[(i + 1):]['userId'].tolist()
                if len(userPool) == 0:
                    return False
                df_dissimilar = dissimilar(user, userPool)
                df_merged = df_result.merge(df_dissimilar, left_on='userId', right_on='userId')
                if not find_all_dissimilar(df_merged):
                    print(f'Could not merge with user {user}')
                else:
                    return True
            return False
    
    if not find_all_dissimilar(dissimilar(user1, userPool)):
        print(f'\n Could not form a group\n')
    

def similarity_values_for_group_formation(userId, userPool, ratings, moviesInCommonMinimum):
    """
    
    """
    targetUserRatings = ratings[ratings['userId'] == userId]

    # get subset of ratings, that only include movies that the target user (userId) has also rated
    userCondition = ratings['userId'].isin(userPool)
    movieCondition = ratings['movieId'].isin(targetUserRatings['movieId'].tolist())
    ratingSubset = ratings[userCondition & movieCondition]

    # filter users that do not have rated more than 'moviesInCommonMinimum' identical movies
    ratingSubsetFiltered = ratingSubset[ratingSubset['userId'].map(ratingSubset['userId'].value_counts()) >= moviesInCommonMinimum]

    if ratingSubsetFiltered.shape[0] == 0:
        correlations = pd.DataFrame(columns=['userId', 'PearsonCorrelation'])
    else:
        # calculate Pearson correlation values
        correlations = similarities.pearson_correlations(targetUserRatings, ratingSubsetFiltered)

    return correlations