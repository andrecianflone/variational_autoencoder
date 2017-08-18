import tensorflow as tf
import numpy as np
from helpers import dense
# See paper: https://arxiv.org/abs/1312.6114

class VAE():
  def __init__(self):
    # Some hyperparams.
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

  def encoder(self, x):
    """ The Encoder encodes the input into the necessary statistics of a latent
    variable model, i.e. a Gaussian in this case. So encode to mean and sigma
    """
    # Input to dense
    h_e = dense(x, self.input_dim, self.h_dim, act=tf.nn.relu, scope="x_to_h1")
    # Dense to latent mean
    z_m = dense(h_e, self.h_dim, self.latent_dim, act=None, scope="lat_mean")
    # Dense to latent sigma
    z_s = dense(h_e, self.h_dim, self.latent_dim, act=None, scope="lat_sigma")
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
    h_d = dense(z, self.latent_dim, self.h_dim, act=tf.nn.relu, scope="z_to_h_d")
    # Hidden to out
    decoded = dense(h_d, self.h_dim, self.input_dim, act=None, scope="h_d_to_out")
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
    # TODO: Sure about this below? Seems different than paper. See Eq 10:
    # https://arxiv.org/abs/1312.6114
    kl_loss = 0.5 * tf.reduce_sum(tf.square(z_mean) + \
            tf.square(z_log_sigma) - tf.log(tf.square(z_log_sigma)) - 1,1)
    return kl_loss

