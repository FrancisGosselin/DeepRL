# Simple Linear Regression

# Importing the libraries
import tensorflow as tf
from tensorflow.layers import batch_normalization
from tensorflow.contrib.layers import fully_connected
from tensorflow.nn import relu
from tensorflow.losses import mean_squared_error
from tensorflow.train import AdamOptimizer
import gym

env = gym.make('MountainCarContinuous-v0')
action_dim = len(env.action_space.sample())
state_dim = len(env.reset())
action_max = env.action_space.high[0]


class Actor():
    
    def __init__(self, state_size, action_size, action_max_value):
        self.gamma = 0.95
        self.tau = 0.001
        self.lr = 0.0001
        self.state_size = state_size
        self.action_size = action_size
        self.action_max_value = action_max_value
        
        
class Critic():
    def __init__(self, state_size, action_size, action_max_value):
        self.gamma = 0.95
        self.tau = 0.001
        self.lr = 0.0001
        self.state_size = state_size
        self.action_size = action_size
        self.action_max_value = action_max_value
        
        self.inputs, self.input_action, self.targets, self.output, self.optimizer = self.create_net()
        
        weights = tf.trainable_variables()
        
        self.inputs_pred, self.input_action_pred , _ , _, _ = self.create_net()

        
    def create_net(self):
        inputs_state = tf.placeholder(dtype='float', shape=[None,4], name='inputs')
        inputs_action = tf.placeholder(dtype='float', shape=[None,4], name='inputs')
        
        targets = tf.placeholder(dtype='float', shape=[None,2], name='targets')
        
        n1 = fully_connected(inputs=inputs_state, num_outputs=32, activation_fn=relu, normalizer_fn=batch_normalization)
        n2 = fully_connected(inputs=n1, num_outputs=32, activation_fn=relu, normalizer_fn=batch_normalization)
        actions = fully_connected(inputs=inputs_action, num_outputs=32, activation_fn=relu, normalizer_fn=batch_normalization)
 
        
        output = relu(tf.add(actions,n2))
        
        loss = mean_squared_error(targets, output)
        
        optimizer = AdamOptimizer().minimize(loss)
         
        return inputs_state, inputs_action, targets, output, optimizer         