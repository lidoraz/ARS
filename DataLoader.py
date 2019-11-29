import pandas as pd
from Constants import CONFIG

"""
Data Loader file, constist a convient loader for all MovieLens datasets.
For each function:
    Input:
    convert_binary - When true, it will convert each rating ranged from 1-5 to 1.
    The logic behind this is that we care only if the has user watched the movie or not.
    Output:
    A tuple of 4:
    df - The rating data frame
    user_item_matrix - A pivot table based on df, represents a relation between each user and movie
    total_users - #users in df
    total_movies - #movies in df 
"""


def get_from_dataset_name(dataset_name, convert_binary):
    print(f'Dataset: {dataset_name} , convert_binary: {convert_binary}')
    if dataset_name == 'movielens100k':
        return get_movielens100k(convert_binary)
    elif dataset_name == 'movielens1m':
        return get_movielens1m(convert_binary)
    elif dataset_name == 'movielens10m':
        return get_movielens10m(convert_binary)
    elif dataset_name == 'movielens20m':
        return get_movielens20m(convert_binary)
    else:
        raise ValueError('Not supported datasets')


def get_movielens100k(convert_binary):
    df = pd.read_csv(CONFIG['MOVIELENS_100k_PATH'], delimiter='\t', header=None,
                     names=['user_id', 'movie_id', 'rating', 'timestamp'])
    return _get_movielens(df, convert_binary)


def get_movielens1m(convert_binary):
    df = pd.read_csv(CONFIG['MOVIELENS_1M_PATH'], delimiter='::', header=None,
                     names=['user_id', 'movie_id', 'rating', 'timestamp'])
    return _get_movielens(df, convert_binary)


def get_movielens10m(convert_binary):
    df = pd.read_csv(CONFIG['MOVIELENS_10M_PATH'], delimiter='::', header=None,
                     names=['user_id', 'movie_id', 'rating', 'timestamp'])
    return _get_movielens(df, convert_binary)


def get_movielens20m(convert_binary):
    df = pd.read_csv(CONFIG['MOVIELENS_20M_PATH'], header=1,
                     names=['user_id', 'movie_id', 'rating', 'timestamp'])
    return _get_movielens(df, convert_binary)


def _get_movielens(df, convert_binary):
    if convert_binary:
        df['rating'] = 1


    return df
