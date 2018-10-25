
"""
TO DO:

"""
import tensorflow as tf
from tensorflow.layers import batch_normalization
from tensorflow.contrib.layers import fully_connected
from tensorflow.nn import relu, tanh
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
        self.state_dim = state_size
        self.action_dim = action_size
        self.action_max_value = action_max_value
        
        self.input_state, self.targets, self.output, self.loss = self.create_net('actor')
        self.input_state_pred, self.input_action_pred , _ , _, _ = self.create_net('actorPred')
        self.weights = tf.trainable_variables('actor')
        self.weights_pred = tf.trainable_variables('actorPred')
        
        self.update_pred_net = [ tf.assign(self.weigts_pred[i], 
                                        tf.add(  tf.multiply(self.tau, self.weights_pred[i]),
                                                 tf.multiply(1.0-self.tau, self.weights_pred[i]) ))
                                for i in range(len(self.weights))]
        self. action_gradient = tf.placeholder('float', shape=[None,self.action_dim])
        tf.gradients(self.output, self.weights, self.action_gradient)
        AdamOptimizer()
        
        
   
    def predict(self, sess, state, action):
        sess.run(self.output, feed_dict = {self.input_state : state} )
        
    def fit(self, sess, state, action, Q_targets):
        sess.run(self.optimizer, feed_dict = { self.input_state : state,
                                                self.targets : Q_targets})
        
    def update_net(self, sess):
        sess.run(self.update_pred_net)
    
    def create_net(self, scope_name):
        inputs_state = tf.placeholder(dtype='float', shape=[None,self.state_dim], name='inputs')        
        targets = tf.placeholder(dtype='float', shape=[None,self.action_dim], name='targets')
        
        with tf.variable_scope(scope_name):
            n1 = fully_connected(inputs=inputs_state, num_outputs=32, activation_fn=relu, normalizer_fn=batch_normalization)
            n2 = fully_connected(inputs=n1, num_outputs=32, activation_fn=relu, normalizer_fn=batch_normalization)
            output = fully_connected(inputs=n2, num_outputs=self.action_dim, activation_fn=tanh, normalizer_fn=batch_normalization)
            output_scaled = tf.scalar_mul(self.action_max_value,output)
            loss = mean_squared_error(targets, output_scaled)
    
        return inputs_state, targets, output_scaled, loss        
    
    
class Critic():
    def __init__(self, state_size, action_size, action_max_value):
        self.gamma = 0.95
        self.tau = 0.001
        self.lr = 0.0001
        self.state_dim = state_size
        self.action_dim = action_size
        self.action_max_value = action_max_value
        
        self.input_state, self.input_action, self.targets, self.output, self.loss = self.create_net('critic')
        self.input_state_pred, self.input_action_pred , _ , _, _ = self.create_net('criticPred')
        weights = tf.trainable_variables('critic')
        weights_pred = tf.trainable_variables('criticPred')
        
        self.update_pred_net = [ tf.assign(self.weigts_pred[i], 
                                        tf.add(  tf.multiply(self.tau,weights_pred[i]),
                                                 tf.multiply(1.0-self.tau, weights_pred[i]) ))
                                for i in range(len(weights))]
        self.AdamOptimizer().minimize(self.loss)
        self.action_gradient = tf.gradient(self.loss, self.input_action)
   
    def predict(self, sess, state, action):
        sess.run(self.output, feed_dict = { self.input_state : state,
                                                self.input_action : action})
        
    def fit(self, sess, state, action, Q_targets):
        sess.run(self.optimizer, feed_dict = { self.input_state : state,
                                                self.input_action : action,
                                                self.targets : Q_targets})
        
    def action_gradient(self, sess, state, action):
        return sess.run(self.action_gradient, feed_dict = { self.input_state : state,
                                                             self.input_action : action})
    def update_net(self, sess):
        sess.run(self.update_pred_net)
    
    def create_net(self, scope_name):
        inputs_state = tf.placeholder(dtype='float', shape=[None,self.state_dim], name='inputs')
        inputs_action = tf.placeholder(dtype='float', shape=[None,self.action_dim], name='inputs')
        
        targets = tf.placeholder(dtype='float', shape=[None,1], name='targets')
        
        with tf.variable_scope(scope_name):
            n1 = fully_connected(inputs=inputs_state, num_outputs=32, activation_fn=relu, normalizer_fn=batch_normalization)
            n2 = fully_connected(inputs=n1, num_outputs=32, activation_fn=relu, normalizer_fn=batch_normalization)
            actions = fully_connected(inputs=inputs_action, num_outputs=32, activation_fn=relu, normalizer_fn=batch_normalization)
            output = relu(tf.add(actions,n2))
            loss = mean_squared_error(targets, output)
        
         
        return inputs_state, inputs_action, targets, output, loss     
  
def train():
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
