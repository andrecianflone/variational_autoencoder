# See original paper for details on theory and formulas:
# https://arxiv.org/abs/1312.6114

# My tensorflow implementation of VAE, using MNIST
# I was inspired by the keras blog by Francois Chollet:
# https://blog.keras.io/building-autoencoders-in-keras.html

import tensorflow as tf
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import norm
from helpers import discrete_cmap
from mlp_vae import VAE

# Get data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
size_train = mnist.train.num_examples # should be 55k

# Params
batch_size = 100
nb_epoch = 5

###############################################################################
# Main stuff
###############################################################################

class Train():
  """ This is just a convenience class for training and plotting
  Keep model separate so easier to understand just the nn part
  """
  def __init__(self):
    # Declare our VAE model
    self.model = VAE()
    num_train_batches = mnist.train.num_examples // batch_size
    self.num_test_batches = mnist.test.num_examples // batch_size
    self.last_batch_loss = None # loss on most recent trained batch

    self.fig = plt.figure(figsize=(6, 6))
    self.ax = self.fig.add_subplot(111)
    # self.ax = self.fig.add_subplot(111, projection='3d')
    ax = plt.axes(xlim=(-12, 12), ylim=(-12, 12))
    self.epoch_time = datetime.now()
    self.cur_epoch_train = 0
    self.batch_callback = True
    plot_n_frames_per_epoch = 60 # number of frames to visualize per epoch
    self.frames_sent_to_plot = 0
    frames = plot_n_frames_per_epoch * nb_epoch -1

    # Check when you need to plot
    if self.batch_callback:
      self.plot_nth_batch = num_train_batches // plot_n_frames_per_epoch
    else:
      self.plot_nth_batch = num_train_batches

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
      Writer = animation.writers['ffmpeg']
      writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)
      animator = animation.FuncAnimation(
              self.fig, self.update, init_func=self.init_plot,
              interval=5, frames=frames, blit=True, repeat=False)
      # animator.save('animation2.mp4', writer=writer)
      # try: animator.to_html5_video()
      # animator.save('animation.mpeg', writer=writer)

      plt.show()
    print('\nframes plotted: ', self.frames_sent_to_plot)

  def init_plot(self):
    """Create base frame
    Return:
      Must return the scatter object so animate knows what to update
    """
    self.increment_training()
    pred, labels = self.get_encoded_test()
    N = len(set(labels.flatten()))
    maxs = np.max(pred, axis=0)
    mins = np.min(pred, axis=0)
    self.scat = plt.scatter(pred[:,0], pred[:,1], c=labels,
                cmap=discrete_cmap(plt, N, 'terrain'))
    plt.colorbar(ticks=range(N))
    plt.clim(-0.5, N - 0.5)
    return [self.scat]

  def update(self, i):
    """ Update the plot after one epoch iteration """
    for i in range(self.plot_nth_batch): self.increment_training();
    pred, labels = self.get_encoded_test()
    self.scat.set_offsets(pred)
    self.scat.set_array(labels)
    self.frames_sent_to_plot += 1
    return [self.scat]

  def get_encoded_test(self):
    """ Returns array of [data_size, latent_size] and labels [data_size,]"""
    pred = np.zeros((mnist.test.num_examples, self.model.latent_dim))
    labels = np.zeros((mnist.test.num_examples))
    for i in range(self.num_test_batches):
      shift = i * batch_size
      batch_xs, batch_ys = mnist.test.next_batch(batch_size)
      fetch = [self.model.z_mean]
      feed = {self.model.x: batch_xs}
      z_mean = self.sess.run(fetch, feed)[0]
      pred[shift:shift+batch_size,:] = z_mean
      labels[shift:shift+batch_size] = batch_ys
    return pred, labels

  def increment_training(self):
    """ Increment training by one batch or one epoch """
    if self.batch_callback: # return after one batch
      self.train_one_batch()
      if mnist.train.epochs_completed > self.cur_epoch_train:
        self.process_end_epoch()
    else: # train a whole epoch before returning
      self.train_one_epoch()

  def train_one_batch(self):
    """ Train only one batch, so can visualize results per batch """
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    fetch = [self.model.optimizer, self.model.recon_loss, self.model.kl_loss, self.model.loss]
    feed={self.model.x: batch_xs}
    _, recon_loss, kl_loss, loss = self.sess.run(fetch, feed)
    self.last_batch_loss = loss
    t2 = datetime.now()
    diff_t = (t2 - self.epoch_time).total_seconds()
    self.print_train_progress(self.cur_epoch_train+1, diff_t,
          loss, self.frames_sent_to_plot, line_end='\r')

  def train_one_epoch(self):
    """ Trains model for a single epoch """
    while mnist.train.epochs_completed == self.cur_epoch_train:
      self.train_one_batch()
    self.process_end_epoch()

  def process_end_epoch(self):
    """ Print some info at the end of an epoch and update epoch counter"""
    self.cur_epoch_train = mnist.train.epochs_completed
    # At end of epoch, print new line
    t2 = datetime.now()
    diff_t = (t2 - self.epoch_time).total_seconds()
    epochs_completed = mnist.train.epochs_completed
    self.print_train_progress(self.cur_epoch_train,
        diff_t, self.last_batch_loss, self.frames_sent_to_plot, line_end="\n")
    self.epoch_time = datetime.now()

  def print_train_progress(self, epoch, diff_t, loss, frames, line_end="\r"):
    """ Nice formatting for trainign info """
    print('epoch: {:2.0f} time: {:>4.1f} | loss: {:>3.4f} | frames plotted: {:>4}'.format(
        epoch, diff_t, loss, frames),end=line_end)

if __name__ == "__main__":
  train = Train()
