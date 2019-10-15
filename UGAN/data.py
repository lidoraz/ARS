import os
import numpy as np
import tensorflow as tf
import tensorlayer as tl

import pandas as pd
## enable debug logging
tl.logging.set_verbosity(tl.logging.DEBUG)


# this function gets two largest 2 integers where their product equals num
def find_max_multipicatns(num):
    for i in range(2, num // 2):  # Iterate from 2 to n / 2
        if (num % i) == 0:
            break
    else:
        raise ValueError('Got prime number')

    for i in range(num):
        for j in range(i):  # j is always lower than i
            if i * j == num:
                return i, j


def get_movielens100k(batch_size):
    import pandas as pd
    import numpy as np


    movielens100k_PATH = 'E:/DEEP_LEARNING/DATA_SETS/ml-100k/u.data'
    df = pd.read_csv(movielens100k_PATH, delimiter='\t', header=None, names=['user_id', 'movie_id', 'rating', 'timestamp']) \
        .drop(columns=['timestamp'])
    util_df=pd.pivot_table(data=df,values='rating',index='user_id',columns='movie_id').fillna(0)

    total_users = util_df.shape[0]
    total_movies = util_df.shape[1]

    # w, h = find_max_multipicatns(total_movies)

    IMAGE_SIZE = 64

    users_data = np.zeros((total_users, 64, 64),dtype=np.double)
    for idx, user_row in enumerate(util_df.iterrows()):
        # user_row = np.zeros((64*64))
        user_row_values = user_row[1].values
        leading_zeros = IMAGE_SIZE**2 - len(user_row_values)
        user_row_values_scaled = np.append(user_row_values, [0] * leading_zeros)
        users_data[idx] = user_row_values_scaled.reshape((IMAGE_SIZE, IMAGE_SIZE))

    # normalize data
    users_data = (users_data - 2.5) / 2.5
    train_ds = tf.data.Dataset.from_tensor_slices(users_data)
    # train_ds = tf.data.Dataset.from_generator(generator_train, output_types=tf.string)
    ds = train_ds.shuffle(buffer_size=4096)
    # ds = ds.shard(num_shards=hvd.size(), index=hvd.rank())
    # ds = ds.repeat(n_epoch)
    # ds = ds.map(_map_fn, num_parallel_calls=4)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=2)

    return ds, users_data.shape[0]

    # return users_data,

def get_celebA(output_size, batch_size):
    # dataset API and augmentation
    CELEBA_PATH = 'E:/DEEP_LEARNING/DATA_SETS/celeba-dataset/img_align_celeba'
    images_path = tl.files.load_file_list(path=CELEBA_PATH, regx='.*.jpg', keep_prefix=True, printable=False)
    def generator_train():
        for image_path in images_path:
            yield image_path.encode('utf-8')
    def _map_fn(image_path):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)  # get RGB with 0~1
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        # image = tf.image.crop_central(image, [FLAGS.output_size, FLAGS.output_size, FLAGS.c_dim])
        # image = tf.image.resize_images(image, FLAGS.output_size])
        image = image[45:173, 25:153, :] # central crop
        image = tf.image.resize([image], (output_size[0], output_size[1]))[0]
        # image = tf.image.crop_and_resize(image, boxes=[[]], crop_size=[64, 64])
        # image = tf.image.resize_image_with_crop_or_pad(image, FLAGS.output_size, FLAGS.output_size) # central crop
        image = tf.image.random_flip_left_right(image)
        image = image * 2 - 1
        return image
    train_ds = tf.data.Dataset.from_generator(generator_train, output_types=tf.string)
    ds = train_ds.shuffle(buffer_size=4096)
    # ds = ds.shard(num_shards=hvd.size(), index=hvd.rank())
    # ds = ds.repeat(n_epoch)
    ds = ds.map(_map_fn, num_parallel_calls=4)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=2)
    return ds, len(images_path)
    # for batch_images in train_ds:
    #     print(batch_images.shape)
    # value = ds.make_one_shot_iterator().get_next()
