import numpy as np
from Gradient_Attack import get_batches

# TODO fake ratings are at start
users = np.array([3, 3, 3, 0, 0, 1, 1, 2, 2])
items = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18])
ratings = np.array([0.5, 0.5, 0.5, 0, 1, 0, 1, 0, 1])
output_dim = 3 # amount of fake ratings
# ------


attack_input = [users, items, ratings]
all_users, all_items, all_rating = attack_input
indexes = np.arange(len(attack_input[0]))
batch_size = 3
for i in range(5):
    # TODO: # add sort by indexes after epoch
    # use np.put(arr, indexes, values)
    # every epoch the values will get updated in attack input, under all_rating column
    for batch in get_batches(np.c_[attack_input.T, indexes], batch_size):
        b_users, b_items, b_ratings, b_indexes = batch[:, 0], batch[:, 1], batch[:, 2], batch[:, 3].astype('int')
        # mask the gradient such that benign user rating will not be changed
        mask = np.zeros((batch_size,))
        mask[np.argwhere(b_indexes < output_dim)] = 1

        batch_rating = np.copy(b_ratings)
        batch_rating[np.logical_and(batch_rating > 0, batch_rating < 1)] = np.random.rand()
        # batch_rating, adv_l, obj_l, reg_l = sess.run([rating_input_out, combined_loss, obj_loss, reg_loss],
        #                                              feed_dict={model.user_inp: b_users,
        #                                                         model.item_inp: b_items,
        #                                                         rating_input: b_ratings,
        #                                                         mask_p: mask,
        #                                                         eps_p: eps})
        batch_rating = batch_rating.reshape((-1,))
        np.put(all_rating, b_indexes, batch_rating)
