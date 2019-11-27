# mode = 0
import os
mode = int(os.environ['RUN_MODE'])
path_prefix = ""
print("Dataset path - Running mode:", mode)
if mode == 0: # default pc work
    path_prefix = 'E:/DEEP_LEARNING/DATA_SETS/'
elif mode == 1: # MacBook
    path_prefix = '/Users/lidora/Google Drive/Code/DATA_SETS_MOVIELENS/'
elif mode == 2: # Ubuntu
    path_prefix = '/media/lidor/48688C7C688C6B10/evo_backup/DEEP_LEARNING/DATA_SETS/'
elif mode == 3: # Colab
    path_prefix = '/content/drive/My Drive/Code/DATA_SETS_MOVIELENS/'
CONFIG = {
    'MOVIELENS_100k_PATH': path_prefix + 'ml-100k/u.data',
    'MOVIELENS_1M_PATH': path_prefix + 'ml-1m/ratings.dat',
    'MOVIELENS_10M_PATH': path_prefix + 'ml-10M100K/ratings.dat',
    'MOVIELENS_20M_PATH': path_prefix + 'ml-20m/ratings.csv'

}


