import pandas as pd
import numpy as np

# this function gets two largest 2 integers where their product equals num
def find_max_multipicatns(num):
    for i in range(2, num // 2):# Iterate from 2 to n / 2
        if (num % i) == 0:
            break
    else:
        raise ValueError('Got prime number')

    for i in range(num):
        for j in range(i): # j is always lower than i
            if i * j == num:
                return i,j


def get_movielens100k(output_size, n_epoch, batch_size):
    movielens100k_PATH = 'E:/DEEP_LEARNING/DATA_SETS/ml-100k/u.data'
    df = pd.read_csv(movielens100k_PATH, delimiter='\t', header=None, names=['user_id', 'movie_id', 'rating', 'timestamp']) \
        .drop(columns=['timestamp'])
    util_df=pd.pivot_table(data=df,values='rating',index='user_id',columns='movie_id').fillna(0)

    total_users = util_df.shape[0]
    total_movies = util_df.shape[1]

    w, h = find_max_multipicatns(total_movies)
    users_data = np.zeros((total_users, w ,h))
    for idx, user_row in enumerate(util_df.iterrows()):
        users_data[idx] = user_row[1].values.reshape(w,h)
    users_data # <class 'tuple'>: (943, 58, 29)
    print('h')




def main():
    get_movielens100k(0,0,0)

if __name__ == '__main__':

    # print(find_max_multipicatns(1682))


    main()