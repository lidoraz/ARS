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
from Constants import SEED

def get_model(num_users, num_items, mf_dim=10, layers=[10], reg_layers=[0], reg_mf=0,  learning_rate=.001, loss_func='binary_crossentropy'):
    import keras
    from keras import initializers
    from keras.regularizers import l1, l2
    from keras.models import Sequential, Model
    from keras.layers import Dense, Lambda, Activation
    from keras.layers import Embedding, Input, Dense, Reshape, Flatten, Dropout

    assert len(layers) == len(reg_layers)
    num_layer = len(layers) #Number of layers in the MLP
    # Input variables
    user_input = Input(shape=(1,), dtype='int32', name='user_input')
    item_input = Input(shape=(1,), dtype='int32', name='item_input')
    
    # Embedding layer
    MF_Embedding_User = Embedding(input_dim = num_users, output_dim = mf_dim, name = 'mf_embedding_user',
                                  embeddings_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=SEED), embeddings_regularizer = l2(reg_mf), input_length=1)
    MF_Embedding_Item = Embedding(input_dim = num_items, output_dim = mf_dim, name = 'mf_embedding_item',
                                  embeddings_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=SEED), embeddings_regularizer = l2(reg_mf), input_length=1)

    MLP_Embedding_User = Embedding(input_dim = num_users, output_dim = int(layers[0]/2), name = "mlp_embedding_user",
                                  embeddings_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=SEED), embeddings_regularizer = l2(reg_layers[0]), input_length=1)
    MLP_Embedding_Item = Embedding(input_dim = num_items, output_dim = int(layers[0]/2), name = 'mlp_embedding_item',
                                  embeddings_initializer =keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=SEED), embeddings_regularizer = l2(reg_layers[0]), input_length=1)
    
    # MF part
    mf_user_latent = Flatten()(MF_Embedding_User(user_input))

    mf_item_latent = Flatten()(MF_Embedding_Item(item_input))
    #mf_vector = merge([mf_user_latent, mf_item_latent], mode = 'mul') # element-wise multiply
    mf_vector = keras.layers.Multiply()([mf_user_latent, mf_item_latent])

    # MLP part 
    mlp_user_latent = Flatten()(MLP_Embedding_User(user_input))
    mlp_item_latent = Flatten()(MLP_Embedding_Item(item_input))
    #mlp_vector = merge([mlp_user_latent, mlp_item_latent], mode = 'concat')
    mlp_vector = keras.layers.Concatenate(axis=-1)([mlp_user_latent, mlp_item_latent])
    for idx in range(1, num_layer):
        layer = Dense(layers[idx], kernel_regularizer = l2(reg_layers[idx]),  bias_regularizer = l2(reg_layers[idx]),  activation='relu', name="layer%d" %idx)
        mlp_vector = layer(mlp_vector)

    # Concatenate MF and MLP parts
    #mf_vector = Lambda(lambda x: x * alpha)(mf_vector)
    #mlp_vector = Lambda(lambda x : x * (1-alpha))(mlp_vector)
    #predict_vector = merge([mf_vector, mlp_vector], mode = 'concat')
    predict_vector = keras.layers.Concatenate(axis=-1)([mf_vector, mlp_vector])
    
    # Final prediction layer
    prediction = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', bias_initializer ='lecun_uniform', name = "prediction")(predict_vector)
    
    model = Model(inputs=[user_input, item_input], 
                  outputs=[prediction])
    from keras.optimizers import Adam
    model.compile(optimizer=Adam(lr=learning_rate), loss=loss_func)
    return model
