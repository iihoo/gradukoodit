import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_satisfaction_dissatisfaction(df_scores):
    df_scores.plot(y=df_scores.columns, xticks=df_scores.index.values, yticks=np.arange(0.0,1.1,step=0.05), xlabel='Round', style='.-')
    plt.hlines(np.arange(0,1.1,step=0.1), 1, len(df_scores.index), linestyles='dotted', color='black')
    plt.legend(loc='best')
    plt.show()