
import tensorflow as tf
from tensorflow.keras import layers



class ActorNetwork(tf.keras.Model):
  """Actor network."""
  def __init__(
      self,
      #alpha,
      # input_dims,
      fc1_dims,
      fc2_dims,
      n_actions,
     # name,
     # chkpt_dir
                 ):

    """Initialize."""
    super().__init__()
    # self.input_dims = tf.keras.layers.Input(input_dims)
    self.d1 = tf.keras.layers.Dense(fc1_dims,activation='relu')
    self.d2 = tf.keras.layers.Dense(fc2_dims,activation='relu')
    self.a = tf.keras.layers.Dense(n_actions ,activation='softmax')

  def call(self, input_data):
    #x = self.inpTensor(input_data)
    x = self.d1(input_data)
    x = self.d2(x)
    a = self.a(x)
    return a
