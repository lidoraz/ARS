'''
Created on Aug 9, 2016
Keras Implementation of Neural Matrix Factorization (NeuMF) recommender model in:
He Xiangnan et al. Neural Collaborative Filtering. In WWW 2017.  

@author: Xiangnan He (xiangnanhe@gmail.com)


Init: HR = 0.1023, NDCG = 0.0463
Iteration 0 [124.0 s]: HR = 0.5974, NDCG = 0.3382, loss = 0.3223 [217.1 s]
The best NeuMF model is saved to Pretrain/ml-1m_NeuMF_8_[64,32,16,8]_1571230994.h5
Iteration 1 [122.6 s]: HR = 0.6356, NDCG = 0.3661, loss = 0.2758 [219.9 s]
The best NeuMF model is saved to Pretrain/ml-1m_NeuMF_8_[64,32,16,8]_1571230994.h5
Iteration 2 [132.0 s]: HR = 0.6465, NDCG = 0.3760, loss = 0.2652 [263.0 s]
The best NeuMF model is saved to Pretrain/ml-1m_NeuMF_8_[64,32,16,8]_1571230994.h5



'''
import numpy as np

# TODO: CRASHED WITH ERROR
# tensorflow.python.framework.errors_impl.ResourceExhaustedError: 2 root error(s) found.
# Resource exhausted:  OOM when allocating tensor with shape[6040,32] and type float on /job:localhost/replica:0/task:0/device:CPU:0 by allocator cpu

# TODO: SOL: https://stackoverflow.com/questions/50694968/resourceexhaustederror-see-above-for-traceback-oom-when-allocating-tensor-wit

import tensorflow.keras
from tensorflow.keras import backend as K
from tensorflow.keras import initializers
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Lambda, Activation
from tensorflow.keras.layers import Embedding, Input, Dense, Reshape, Flatten, Dropout
from tensorflow.keras.optimizers import Adagrad, Adam, SGD, RMSprop
# from evaluate import evaluate_model
# from Dataset import Dataset
from time import time
import sys
# import GMF, MLP
import argparse

#################### Arguments ####################
from Data import Data
from DataLoader import get_from_dataset_name


def parse_args():
    parser = argparse.ArgumentParser(description="Run NeuMF.")
    parser.add_argument('--path', nargs='?', default='Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ml-1m',
                        help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--num_factors', type=int, default=8,
                        help='Embedding size of MF model.')
    parser.add_argument('--layers', nargs='?', default='[64,32,16,8]',
                        help="MLP layers. Note that the first layer is the concatenation of user and item embeddings. So layers[0]/2 is the embedding size.")
    parser.add_argument('--reg_mf', type=float, default=0,
                        help='Regularization for MF embeddings.')                    
    parser.add_argument('--reg_layers', nargs='?', default='[0,0,0,0]',
                        help="Regularization for each MLP layer. reg_layers[0] is the regularization for embeddings.")
    parser.add_argument('--num_neg', type=int, default=4,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--learner', nargs='?', default='adam',
                        help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show performance per X iterations')
    parser.add_argument('--out', type=int, default=1,
                        help='Whether to save the trained model.')
    parser.add_argument('--mf_pretrain', nargs='?', default='',
                        help='Specify the pretrain model file for MF part. If empty, no pretrain will be used')
    parser.add_argument('--mlp_pretrain', nargs='?', default='',
                        help='Specify the pretrain model file for MLP part. If empty, no pretrain will be used')
    return parser.parse_args()

def init_normal(shape, name=None):
    return initializers.normal(shape, scale=0.01, name=name)

def get_model(num_users, num_items, mf_dim=10, layers=[10], reg_layers=[0], reg_mf=0):
    assert len(layers) == len(reg_layers)
    num_layer = len(layers) #Number of layers in the MLP
    # Input variables
    user_input = Input(shape=(1,), dtype='int32', name = 'user_input')
    item_input = Input(shape=(1,), dtype='int32', name = 'item_input')
    
    # Embedding layer
    MF_Embedding_User = Embedding(input_dim = num_users, output_dim = mf_dim, name = 'mf_embedding_user',
                                  embeddings_initializer = tensorflow.keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None), embeddings_regularizer = l2(reg_mf), input_length=1)
    MF_Embedding_Item = Embedding(input_dim = num_items, output_dim = mf_dim, name = 'mf_embedding_item',
                                  embeddings_initializer = tensorflow.keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None), embeddings_regularizer = l2(reg_mf), input_length=1)

    MLP_Embedding_User = Embedding(input_dim = num_users, output_dim = int(layers[0]/2), name = "mlp_embedding_user",
                                  embeddings_initializer = tensorflow.keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None), embeddings_regularizer = l2(reg_layers[0]), input_length=1)
    MLP_Embedding_Item = Embedding(input_dim = num_items, output_dim = int(layers[0]/2), name = 'mlp_embedding_item',
                                  embeddings_initializer =tensorflow.keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None), embeddings_regularizer = l2(reg_layers[0]), input_length=1)
    
    # MF part
    mf_user_latent = Flatten()(MF_Embedding_User(user_input))
    mf_item_latent = Flatten()(MF_Embedding_Item(item_input))
    #mf_vector = merge([mf_user_latent, mf_item_latent], mode = 'mul') # element-wise multiply
    mf_vector = tensorflow.keras.layers.Multiply()([mf_user_latent, mf_item_latent])

    # MLP part 
    mlp_user_latent = Flatten()(MLP_Embedding_User(user_input))
    mlp_item_latent = Flatten()(MLP_Embedding_Item(item_input))
    #mlp_vector = merge([mlp_user_latent, mlp_item_latent], mode = 'concat')
    mlp_vector = tensorflow.keras.layers.Concatenate(axis=-1)([mlp_user_latent, mlp_item_latent])
    for idx in range(1, num_layer):
        layer = Dense(layers[idx], kernel_regularizer = l2(reg_layers[idx]),  bias_regularizer = l2(reg_layers[idx]),  activation='relu', name="layer%d" %idx)
        mlp_vector = layer(mlp_vector)

    # Concatenate MF and MLP parts
    #mf_vector = Lambda(lambda x: x * alpha)(mf_vector)
    #mlp_vector = Lambda(lambda x : x * (1-alpha))(mlp_vector)
    #predict_vector = merge([mf_vector, mlp_vector], mode = 'concat')
    predict_vector = tensorflow.keras.layers.Concatenate(axis=-1)([mf_vector, mlp_vector])
    
    # Final prediction layer
    prediction = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', bias_initializer ='lecun_uniform', name = "prediction")(predict_vector)
    
    model = Model(inputs=[user_input, item_input], 
                  outputs=[prediction])
    
    return model

def get_train_instances(train, num_negatives):
    user_input, item_input, labels = [],[],[]
    num_users = train.shape[0]
    for (u, i) in train.keys():
        # positive instance
        user_input.append(u)
        item_input.append(i)
        labels.append(1)
        # negative instances
        for t in range(num_negatives):
            j = np.random.randint(num_items)
            while (u, j) in train: #train.has_key((u, j)):
                j = np.random.randint(num_items)
            user_input.append(u)
            item_input.append(j)
            labels.append(0)
    return user_input, item_input, labels



from Evalute import evaluate_model


if __name__ == '__main__':
    args = parse_args()
    num_epochs = args.epochs
    batch_size = args.batch_size
    mf_dim = args.num_factors
    layers = eval(args.layers)
    reg_mf = args.reg_mf
    reg_layers = eval(args.reg_layers)
    num_negatives = args.num_neg
    learning_rate = args.lr
    learner = args.learner
    verbose = args.verbose
    mf_pretrain = args.mf_pretrain
    mlp_pretrain = args.mlp_pretrain
            
    topK = 10
    evaluation_threads = 1#mp.cpu_count()
    print("NeuMF arguments: %s " %(args))
    model_out_file = 'Pretrain/%s_NeuMF_%d_%s_%d.h5' %(args.dataset, mf_dim, args.layers, time())

    # Loading data
    t1 = time()
    df, user_item_matrix, total_users, total_movies = get_from_dataset_name('movielens1m')
    data = Data(df, user_item_matrix, total_users, total_movies)
    df_removed_recents = data.filter_trainset()

    # dataset = Dataset(args.path + args.dataset)
    # train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    num_users, num_items = total_users, total_movies
    # print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d"
    #       %(time()-t1, num_users, num_items, train.nnz, len(testRatings)))
    
    # Build model
    model = get_model(num_users, num_items, mf_dim, layers, reg_layers, reg_mf)
    model.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy')
    print('got_model done')
        
    # Init performance
    testset_percentage = 0.5
    test_set = data.create_testset(percent=testset_percentage)
    (hits, ndcgs) = evaluate_model(model, test_set)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    print('Init: HR = %.4f, NDCG = %.4f' % (hr, ndcg))
    best_hr, best_ndcg, best_iter = hr, ndcg, -1


    #todo added:
    TRAINING = True

    if TRAINING:
        if args.out > 0:
            model.save_weights(model_out_file, overwrite=True)
            # Training model
        for epoch in range(num_epochs):
            t1 = time()
            # Generate training instances
            user_input, item_input, labels = get_train_instances(train, num_negatives)

            # Training
            hist = model.fit([np.array(user_input), np.array(item_input)], #input
                             np.array(labels), # labels
                             batch_size=batch_size, epochs=1, verbose=0, shuffle=True)
            t2 = time()

            # Evaluation
            if epoch %verbose == 0:
                (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
                hr, ndcg, loss = np.array(hits).mean(), np.array(ndcgs).mean(), hist.history['loss'][0]
                print('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]'
                      % (epoch,  t2-t1, hr, ndcg, loss, time()-t2))
                if hr > best_hr:
                    best_hr, best_ndcg, best_iter = hr, ndcg, epoch
                    if args.out > 0:
                        model.save_weights(model_out_file, overwrite=True)
                        print("The best NeuMF model is saved to %s" % (model_out_file))
    else:

        model_out_file = 'Pretrain/ml-1m_NeuMF_8_[64,32,16,8]_1571090710.h5'
        print('Loading from:', model_out_file)
        model.load_weights(model_out_file)
        epoch = 0
        t2 = time()
        (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)

        hr, ndcg, loss = np.array(hits).mean(), np.array(ndcgs).mean(), 0
        print('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]'
              % (epoch, t2 - t1, hr, ndcg, loss, time() - t2))


