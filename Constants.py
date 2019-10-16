mode = 0
path_prefix = ""
print("Dataset path - Running mode:", mode)
if mode == 0: # default pc work
    path_prefix = 'E:/DEEP_LEARNING/DATA_SETS/'
elif mode == 1: # macbook
    path_prefix = '/Users/lidora/Google Drive/Code/DATA_SETS_MOVIELENS'


CONFIG = {
    'MOVIELENS_100k_PATH': path_prefix + 'ml-100k/u.data',
    'MOVIELENS_1M_PATH': path_prefix + 'ml-1m/ratings.dat',
    'MOVIELENS_10M_PATH': path_prefix + 'ml-10M100K/ratings.dat',
    'MOVIELENS_20M_PATH': path_prefix + 'ml-20m/ratings.csv'

}


