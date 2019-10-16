import os, time, multiprocessing
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from glob import glob
from data import get_celebA, get_movielens100k
from model import get_generator, get_discriminator

class FLAGS(object):
    def __init__(self, n_epoch=25, z_dim=100, lr=0.0002, beta1=0.5, batch_size=64,
                 output_width=64, output_height= 64 , sample_size=64, c_dim=3, save_every_epoch=1):
        self.n_epoch = n_epoch # "Epoch to train [25]"
        self.z_dim = z_dim # "Num of noise value]"
        self.lr = lr # "Learning rate of for adam [0.0002]")
        self.beta1 = beta1 # "Momentum term of adam [0.5]")
        self.batch_size = batch_size # "The number of batch images [64]")
        self.output_width = output_width # "The size of the output images to produce [64]")
        self.output_height = output_height
        self.sample_size = sample_size # "The number of sample images [64]")
        self.c_dim = c_dim # "Number of image channels. [3]")
        self.save_every_epoch = save_every_epoch # "The interval of saveing checkpoints.")
        # self.dataset = "celebA" # "The name of dataset [celebA, mnist, lsun]")
        self.checkpoint_dir = "checkpoint" # "Directory name to save the checkpoints [checkpoint]")
        self.sample_dir = "samples" # "Directory name to save the image samples [samples]")
        assert np.sqrt(self.sample_size) % 1 == 0., 'Flag `sample_size` needs to be a perfect square'

flags = FLAGS(c_dim=1)
num_tiles = int(np.sqrt(flags.sample_size))

tl.files.exists_or_mkdir(flags.checkpoint_dir) # save model
tl.files.exists_or_mkdir(flags.sample_dir) # save generated image

def train():

    DATASET = 'MOVIE_LENS'
    # DATASET = 'CELEBA'


    print("DATASET:", DATASET)
    if DATASET == 'CELEBA':
        images, num_examples = get_celebA((64, 64), flags.batch_size)
        b_w = False
    elif DATASET == 'MOVIE_LENS':
        images, num_examples = get_movielens100k(flags.batch_size)
        b_w = True
        flags.c_dim = 1
        # flags.output_height = images._flat_shapes[0][1]
        # flags.output_width = images._flat_shapes[0][2]
    else:
        raise(ValueError('invalid DATAET param'))

    G = get_generator([None, flags.z_dim], b_w= b_w)
    D = get_discriminator([None, flags.output_width, flags.output_height, flags.c_dim])

    G.train()
    D.train()

    d_optimizer = tf.optimizers.Adam(flags.lr, beta_1=flags.beta1)
    g_optimizer = tf.optimizers.Adam(flags.lr, beta_1=flags.beta1)

    n_step_epoch = int(num_examples // flags.batch_size)
    
    # Z = tf.distributions.Normal(0., 1.)
    for epoch in range(flags.n_epoch):
        for step, batch_images in enumerate(images):
            if batch_images.shape[0] != flags.batch_size: # if the remaining data in this epoch < batch_size
                break
            step_time = time.time()
            with tf.GradientTape(persistent=True) as tape:
                # z = Z.sample([flags.batch_size, flags.z_dim]) 
                z = np.random.normal(loc=0.0, scale=1.0, size=[flags.batch_size, flags.z_dim]).astype(np.float32)
                d_logits = D(G(z))
                d2_logits = D(batch_images)
                # discriminator: real images are labelled as 1
                d_loss_real = tl.cost.sigmoid_cross_entropy(d2_logits, tf.ones_like(d2_logits), name='dreal')
                # discriminator: images from generator (fake) are labelled as 0
                d_loss_fake = tl.cost.sigmoid_cross_entropy(d_logits, tf.zeros_like(d_logits), name='dfake')
                # combined loss for updating discriminator
                d_loss = d_loss_real + d_loss_fake
                # generator: try to fool discriminator to output 1
                g_loss = tl.cost.sigmoid_cross_entropy(d_logits, tf.ones_like(d_logits), name='gfake')

            grad = tape.gradient(g_loss, G.trainable_weights)
            g_optimizer.apply_gradients(zip(grad, G.trainable_weights))
            grad = tape.gradient(d_loss, D.trainable_weights)
            d_optimizer.apply_gradients(zip(grad, D.trainable_weights))
            del tape

            print("Epoch: [{}/{}] [{}/{}] took: {:.3f}, d_loss: {:.5f}, g_loss: {:.5f}".format(epoch, \
                  flags.n_epoch, step, n_step_epoch, time.time()-step_time, d_loss, g_loss))
        
        if np.mod(epoch, flags.save_every_epoch) == 0:
            G.save_weights('{}/G.npz'.format(flags.checkpoint_dir), format='npz')
            D.save_weights('{}/D.npz'.format(flags.checkpoint_dir), format='npz')
            G.eval()
            result = G(z)
            G.train()
            tl.visualize.save_images(result.numpy(), [num_tiles, num_tiles], '{}/train_{:02d}.png'.format(flags.sample_dir, epoch))

if __name__ == '__main__':
    train()
