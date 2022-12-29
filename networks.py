
import tensorflow as tf
from tensorflow.keras import layers



class CriticNetwork(tf.keras.Model):
  """Actor network."""
  def __init__(self,
        #    beta,
            # input_dims,
            fc1_dims,
            fc2_dims,
            # n_agents,
        #    n_actions,
        # name, chkpt_dir
                 ):

    """Initialize."""
    super().__init__()
    # self.input_dims = tf.keras.layers.Input(input_dims)
    self.d1 = tf.keras.layers.Dense(fc1_dims,activation='relu')
    self.d2 = tf.keras.layers.Dense(fc2_dims,activation='relu')
    self.a = tf.keras.layers.Dense(1 ,activation='linear')

  def call(self, input_data,action):
    input_data = tf.concat([input_data, action],  axis=1)
    x = self.d1(input_data)
    x = self.d2(x)
    a = self.a(x)
    return a

  def save_checkpoint(self):
    T.save(self.state_dict(), self.chkpt_file)

  def load_checkpoint(self):
    self.load_state_dict(T.load(self.chkpt_file))




class ActorNetwork(tf.keras.Model):
  """Actor network."""
  def __init__(
      self,
      #alpha,
      # input_dims,
      fc1_dims,
      fc2_dims,
      n_actions,
      name,
      chkpt_dir
                 ):

    """Initialize."""
    super().__init__()
    self.d1 = tf.keras.layers.Dense(fc1_dims,activation='relu')
    self.d2 = tf.keras.layers.Dense(fc2_dims,activation='relu')
    self.a = tf.keras.layers.Dense(n_actions ,activation='softmax')

  def call(self, input_data):
    x = self.d1(input_data)
    x = self.d2(x)
    a = self.a(x)
    return a


  def save_checkpoint(self):
    T.save(self.state_dict(), self.chkpt_file)

  def load_checkpoint(self):
    self.load_state_dict(T.load(self.chkpt_file))
