import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_satisfaction_dissatisfaction(df_scores):
    df_scores.plot(y=df_scores.columns, xticks=df_scores.index.values, yticks=np.arange(0.0,1.1,step=0.05), xlabel='Round', style='.-')
    plt.hlines(np.arange(0,1.1,step=0.1), 1, len(df_scores.index), linestyles='dotted', color='black')
    plt.legend(loc='best')
    plt.show()


def plot_results(df_results):
    """
    Plot 9 (3x3) subplots from the results dataframe.
    
    Plots values from columns 'GroupSatO:HYBRID', 'GroupSatO:MODIF.AGGR.', 'GroupDisO:HYBRID', 'GroupDisO:MODIF.AGGR.'.
    """
    
    fig, axes = plt.subplots(nrows=3, ncols=3)
    idx = pd.IndexSlice
    for groupIndex in range(1, 10):
        x = 0
        if (groupIndex in [2, 5, 8]):
            x = 1
        elif (groupIndex in [3, 6, 9]):
            x = 2
        y = 0
        if (groupIndex in [4, 5, 6]):
            y = 1
        elif (groupIndex in [7, 8, 9]):
            y = 2
        df_plot = df_results.loc[idx[groupIndex, :]][['GroupSatO:HYBRID', 'GroupSatO:MODIF.AGGR.', 'GroupDisO:HYBRID', 'GroupDisO:MODIF.AGGR.']]
        df_plot.plot(ax=axes[x, y], xticks=df_plot.index.values)
    plt.show()