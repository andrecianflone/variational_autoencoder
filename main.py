# See original paper for details on theory and formulas:
# https://arxiv.org/abs/1312.6114

# My tensorflow implementation of VAE, using MNIST
# Some ideas like the plotting is from Francois Chollet blog:
# https://blog.keras.io/building-autoencoders-in-keras.html

import tensorflow as tf
import numpy as np
from datetime import datetime
from tensorflow.contrib.layers import xavier_initializer as glorot
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
Writer = animation.writers['ffmpeg']
writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)
from scipy.stats import norm

# Get data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
size_train = mnist.train.num_examples # should be 55k

# Params
batch_size = 100
nb_epoch = 5

class VAE():
  def __init__(self):
    self.input_dim = 784
    self.h_dim = 256 # hidden layer dimension
    self.latent_dim = 2 # size of latent state hidden layer
    self.data_type = tf.float32

    # The MNIST inputs
    self.x = tf.placeholder(self.data_type, shape=[None, self.input_dim])
    self.batch_size = tf.shape(self.x)[0]

    # The VAE
    self.z_mean, z_log_sigma = self.encoder(self.x)
    z = self.sampler(self.z_mean, z_log_sigma)
    self.decoded = self.decoder(z)

    # Loss and optimizer
    self.recon_loss = self._recon_loss(self.x, self.decoded)
    self.kl_loss = self._kl_loss(self.z_mean, z_log_sigma)

    self.loss = tf.reduce_mean(self.recon_loss + self.kl_loss)
    self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.loss)

  def fc(self, x, in_dim, out_dim, scope, act=None):
    """ Fully connected layer builder"""
    with tf.variable_scope(scope):
      weights = tf.get_variable("weights", shape=[in_dim, out_dim],
                dtype=self.data_type, initializer=glorot())
      biases = tf.get_variable("biases", out_dim,
                dtype=self.data_type, initializer=tf.constant_initializer(0.0))
      # Pre activation
      h = tf.matmul(x,weights) + biases
      # Post activation
      if act:
        h = act(h)
      return h

  def encoder(self, x):
    """ The Encoder encodes the input into the necessary statistics of a latent
    variable model, i.e. a Gaussian in this case. So encode to mean and sigma
    """
    # Input to dense
    h_e = self.fc(x, self.input_dim, self.h_dim, act=tf.nn.relu, scope="x_to_h1")
    # Dense to latent mean
    z_m = self.fc(h_e, self.h_dim, self.latent_dim, act=None, scope="lat_mean")
    # Dense to latent sigma
    z_s = self.fc(h_e, self.h_dim, self.latent_dim, act=None, scope="lat_sigma")
    return z_m, z_s

  def sampler(self, z_mean, z_log_sigma):
    """ We sample a datapoint from a Gaussian with the reparameterization trick
    The encoder gives us a Gaussian mean and stdev. We add random noise and this
    gives us a sample from the Gaussian density
    """
    epsilon = tf.random_normal(\
        [self.batch_size, self.latent_dim], 0,1,dtype=self.data_type)
    z = z_mean + tf.exp(z_log_sigma)*epsilon
    return z

  def decoder(self, z):
    """ The decoder maps the sampled datapoint to the output,
    which is the original input)
    """
    # Latent to hidden
    h_d = self.fc(z, self.latent_dim, self.h_dim, act=tf.nn.relu, scope="z_to_h_d")
    # Hidden to out
    decoded = self.fc(h_d, self.h_dim, self.input_dim, act=None, scope="h_d_to_out")
    return decoded

  def _recon_loss(self, x, decoded):
    """Loss between reconstructued input and true input"""
    # recon_loss is shape [batch_size,]
    recon_loss = tf.reduce_sum(\
            tf.nn.sigmoid_cross_entropy_with_logits(\
            labels=x, logits=decoded, name="recon_loss"), axis=1) # sum rows
    return recon_loss

  def _kl_loss(self, z_mean, z_log_sigma):
    """ KL divergence loss"""
    # kl_loss is shape [batch_size]
    kl_loss = 0.5 * tf.reduce_sum(tf.square(z_mean) + \
            tf.square(z_log_sigma) - tf.log(tf.square(z_log_sigma)) - 1,1)
    return kl_loss

###############################################################################
# Main stuff
###############################################################################

class Train():
  """ Convenience class for training and plotting """
  def __init__(self):
    # Declare our VAE model
    self.model = VAE()
    num_train_batches = mnist.train.num_examples // batch_size
    self.last_batch_loss = None # loss on most recent trained batch

    self.fig = plt.figure(figsize=(6, 6))
    self.ax = self.fig.add_subplot(111)
    # self.ax = self.fig.add_subplot(111, projection='3d')
    ax = plt.axes(xlim=(-12, 12), ylim=(-12, 12))
    self.epoch_time = datetime.now()
    self.cur_epoch_train = 0
    batch_callback = True
    self.data_stream = self.incr_train_yield_test_pred(batch_callback=batch_callback)
    plot_n_frames_per_epoch = 30 # number of frames to visualize per epoch
    self.plot_every_nth_frame = num_train_batches // plot_n_frames_per_epoch

    if batch_callback:
      frames = nb_epoch*self.num_train_batches-2
    else:
      frames = nb_epoch - 2

    # Context manager for convenience
    # Uses default graph
    with tf.Session() as sess:
      self.sess = sess
      tf.global_variables_initializer().run()
      ## TRAIN
      print('*'*79)
      print('Training VAE')
      print('*'*79)

      # Training and testing is managed by matplotlib iterator
      animator = animation.FuncAnimation(
              self.fig, self.update_plot, init_func=self.init_plot,
              interval=100, frames=frames, blit=True, repeat=False)
      # animator.save('animation.mp4', writer=writer)
      # animator.save('animation.mpeg', writer=writer)
      plt.show()

  def init_plot(self):
    """Create base frame
    Return:
      Must return the scatter object so animate knows what to update
    """
    pred, labels = next(self.data_stream)
    maxs = np.max(pred, axis=0)
    mins = np.min(pred, axis=0)
    # self.ax.set_xlim(mins[0], maxs[0])
    # self.ax.set_xlim(mins[1], maxs[1])
    # self.ax.set_zlim(mins[2], maxs[2])
    self.scat = plt.scatter(pred[:,0], pred[:,1], c=labels)
    plt.colorbar()
    return [self.scat]

  def update_plot(self, i):
    """ Update the plot after one epoch iteration """
    pred, labels = next(self.data_stream)
    self.scat.set_offsets(pred)
    self.scat.set_array(labels)
    return [self.scat]

  def get_encoded_test(self):
    """ Returns array of [data_size, latent_size] and labels [data_size,]"""
    pred = np.zeros((mnist.test.num_examples, self.model.latent_dim))
    labels = np.zeros((mnist.test.num_examples))
    for i in range(self.num_test_batches):
      shift = i * batch_size
      batch_xs, batch_ys = mnist.train.next_batch(batch_size)
      fetch = [self.model.z_mean]
      feed = {self.model.x: batch_xs}
      z_mean = self.sess.run(fetch, feed)[0]
      pred[shift:shift+batch_size,:] = z_mean
      labels[shift:shift+batch_size] = batch_ys
    return pred, labels

  def incr_train_yield_test_pred(self, batch_callback=False):
    """ Yields predictions and labels of test set after an epoch or batch """
    while True: # the animator decides when to stop the generator
      if batch_callback: # yield after one batch
        self.train_one_batch()
        if mnist.train.epochs_completed > self.cur_epoch_train:
          self.process_end_epoch()
      else: # train a whole epoch before yielding
        self.train_one_epoch()
      pred, labels = self.get_encoded_test()
      yield pred, labels

  def train_one_batch(self):
    """ Train only one batch, so can visualize results per batch """
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    fetch = [self.model.optimizer, self.model.recon_loss, self.model.kl_loss, self.model.loss]
    feed={self.model.x: batch_xs}
    _, recon_loss, kl_loss, loss = self.sess.run(fetch, feed)
    self.last_batch_loss = loss
    t2 = datetime.now()
    diff_t = (t2 - self.epoch_time).total_seconds()
    print('epoch: {:2.0f} time: {:2.1f} | loss: {:.4f}'.format(
        self.cur_epoch_train, diff_t, loss), end='\r')

  def train_one_epoch(self):
    """ Trains model for a single epoch """
    while mnist.train.epochs_completed == self.cur_epoch_train:
      self.train_one_batch()
    self.process_end_epoch()

  def process_end_epoch(self):
    """ Print some info at the end of an epoch """
    self.cur_epoch_train = mnist.train.epochs_completed
    # At end of epoch, print new line
    t2 = datetime.now()
    diff_t = (t2 - self.epoch_time).total_seconds()
    epochs_completed = mnist.train.epochs_completed
    print('epoch: {:2.0f} time: {:2.1f} | loss: {:.4f}'.format(
      self.cur_epoch_train, diff_t, self.last_batch_loss), end="\n")
    self.epoch_time = datetime.now()

train = Train()
