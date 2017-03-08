import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import xavier_initializer as glorot

data_type = tf.float32

def dense(x, in_dim, out_dim, scope, act=None):
  """ Fully connected layer builder"""
  with tf.variable_scope(scope):
    weights = tf.get_variable("weights", shape=[in_dim, out_dim],
              dtype=data_type, initializer=glorot())
    biases = tf.get_variable("biases", out_dim,
              dtype=data_type, initializer=tf.constant_initializer(0.0))
    # Pre activation
    h = tf.matmul(x,weights) + biases
    # Post activation
    if act:
      h = act(h)
    return h

def discrete_cmap(plt, N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map
    Thanks to Jake VanderPlas, from:
    https://gist.github.com/jakevdp/91077b0cae40f8f8244a
    Sample usage:
      plt.scatter(x, y, c=c, s=50, cmap=discrete_cmap(N, 'cubehelix'))
      plt.colorbar(ticks=range(N))
      plt.clim(-0.5, N - 0.5)
      plt.show()
    """

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:
    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)

