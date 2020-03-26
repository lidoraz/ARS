import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# from Gradient_Attack import DATASET_NAME
def plot_interactions_neg(training_set, DATASET_NAME, atleast= None, title=''):
    df = pd.DataFrame({'u': training_set[0], 'i': training_set[1], 'r': training_set[2]})
    return plot_interations(df[df.r == 1], DATASET_NAME, atleast, title)

def plot_interations(df, DATASET_NAME, atleast= None, title=''):
    """
    :param data: (i,u, r) array like
    :param limit: limit will plot only #limit from both ends
    :return:
    """
    summed = df[['i', 'r']].groupby('i').sum()
    if atleast:
        summed = summed[summed.r >= atleast]
        plt.ylabel(f'Interactions (atleast {atleast})')
    plt.bar(summed.r.index, height=summed.r.values)
    # plt.xticks(summed.r.index)
    plt.grid(True)
    plt.title(f'T={title} interactions={len(df)}')
    plt.xlabel('Item index')
    if DATASET_NAME == 'movielens1m':
        plt.xlim([0, 3710])
    elif DATASET_NAME =='movielens100k':
        plt.xlim([0, 1700])
    plt.savefig(f'ga_{DATASET_NAME}_t{title}.png')
    plt.show()

# n_samples = 100
# u = np.random.randint(1, 10, n_samples)
# i = np.random.randint(1, 1600, n_samples)
# r = np.ones(n_samples)
#
# data = (u, i , r)
#
# plot_interations(data)
