import pandas as pd
import random

import similarities

# create a group type 3 + 2
def create_group_type_3_2(df_ratings, moviesInCommonMinimum, similarityThreshold, dissimilarityThreshold):
    """
    Create a group with two subgroups.
    
    Users in a subgroup are similar with each other, but dissimilar to all users in other subgroups.
    
    Users are considered similar when they have a Pearson Correlation value above 'similarityThreshold', and dissimilar when they have a Pearson Correlation value below 'dissimilarityThreshold'.
    NOTE no need to rename PearsonCorrelation columns??
    NOTE if any of the merged dataframes are empty, a group cannot be formed
    NOTE if there are no similar or no dissimilar users for any group member, a group cannot be formed
    """
    group = []

    # pick the first user and add to group
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
