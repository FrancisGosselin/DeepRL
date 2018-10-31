
"""
TO DO:

"""
import tensorflow as tf
from tensorflow.layers import batch_normalization
from tensorflow.contrib.layers import fully_connected, batch_norm
from tensorflow.nn import relu, tanh
from tensorflow.losses import mean_squared_error
from tensorflow.train import AdamOptimizer
from exploration import OrnsteinUhlenbeckActionNoise
from collections import deque
import numpy as np
import random
import gym

class Actor():
    def __init__(self, state_size, action_size, action_max_value, batch_size):
        self.gamma = 0.95
        self.tau = 0.001
        self.lr = 0.0001
        self.state_dim = state_size
        self.action_dim = action_size
        self.action_max_value = action_max_value
        
        self.input_state, self.targets, self.output, self.loss = self.create_net('actor')
        self.input_state_pred, self.input_action_pred , _ , _ = self.create_net('pred_net_actor')
        self.weights = tf.trainable_variables('actor')
        self.weights_pred = tf.trainable_variables('pred_net')

        self.update_pred_net = [ tf.assign(self.weights_pred[i], 
                                        tf.add(  tf.multiply(self.tau, self.weights[i]),
                                                 tf.multiply(1.0-self.tau, self.weights_pred[i]) ))
                                for i in range(len(self.weights))]
        
        self.action_gradient = tf.placeholder('float', shape=[None,self.action_dim])
        self.J_gradient = tf.gradients(self.output, self.weights, -self.action_gradient)
        self.J_gradient_normalized = list(map(lambda x: x/batch_size, self.J_gradient))
        self.optimizer = AdamOptimizer().apply_gradients(zip(self.J_gradient_normalized, self.weights))
        
        
   
    def predict(self, sess, state):
        return sess.run(self.output, feed_dict = {self.input_state : state} )
        
    def fit(self, sess, state, action_targets, action_gradient):
        sess.run(self.optimizer, feed_dict = { self.input_state : state,
                                               self.targets : action_targets,
                                               self.action_gradient : action_gradient})
        
    def update_net(self, sess):
        sess.run(self.update_pred_net)
    
    def create_net(self, scope_name):
        inputs_state = tf.placeholder(dtype='float', shape=[None,self.state_dim], name='inputs')        
        targets = tf.placeholder(dtype='float', shape=[None,self.action_dim], name='targets')
        
        with tf.variable_scope(scope_name, reuse = tf.AUTO_REUSE):
            n1 = fully_connected(inputs=inputs_state, num_outputs=32, activation_fn=relu)
            n2 = fully_connected(inputs=n1, num_outputs=32, activation_fn=relu)
            output = fully_connected(inputs=n2, num_outputs=self.action_dim, activation_fn=tanh)
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
        self.input_state_pred, self.input_action_pred , _ , _, _ = self.create_net('pred_net_critic')
        self.weights = tf.trainable_variables('critic')
        self.weights_pred = tf.trainable_variables('pred_net')
        
        self.update_pred_net = [ tf.assign(self.weights_pred[i], 
                                        tf.add(  tf.multiply(self.tau,self.weights[i]),
                                                 tf.multiply(1.0-self.tau, self.weights_pred[i]) ))
                                for i in range(len(self.weights))]
        self.optimizer = AdamOptimizer().minimize(self.loss)
        self.action_gradient = tf.gradients(self.loss, self.input_action)
   
    def predict(self, sess, state, action):
        return sess.run(self.output, feed_dict = { self.input_state : state,
                                                self.input_action : action})
        
    def fit(self, sess, state, action, Q_target):
        sess.run(self.optimizer, feed_dict = { self.input_state : state,
                                                self.input_action : action,
                                                self.targets : Q_target})
        
    def get_action_gradient(self, sess, state, action, targets):
        return sess.run(self.action_gradient, feed_dict = { self.input_state : state,
                                                             self.input_action : action,
                                                            self.targets: targets})
    def update_net(self, sess):
        sess.run(self.update_pred_net)
    
    def create_net(self, scope_name):
        inputs_state = tf.placeholder(dtype='float', shape=[None,self.state_dim], name='inputs')
        inputs_action = tf.placeholder(dtype='float', shape=[None,self.action_dim], name='inputs')
        
        targets = tf.placeholder(dtype='float', shape=[None,1], name='targets')
        
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            n1 = fully_connected(inputs=inputs_state, num_outputs=32, activation_fn=relu)
            n2 = fully_connected(inputs=n1, num_outputs=1, activation_fn=relu)
            actions = fully_connected(inputs=inputs_action, num_outputs=1, activation_fn=relu)
            output = relu(tf.add(actions,n2))
            loss = mean_squared_error(targets, output)
        
         
        return inputs_state, inputs_action, targets, output, loss 
    
def fit_batch(sess, batch, actor, critic, discount):
    Q_targets = []
    action_gradients = []
    for i in range(len(batch)):
        next_state, reward, action, state, done = batch[i]
        next_action = actor.predict(sess,next_state)
        Q_target = reward 
        if(not done):
            Q_target += discount*critic.predict(sess, next_state, next_action)[0][0]
        Q_targets.append([Q_target])
        action_gradient = critic.get_action_gradient(sess, state, action, [[Q_target]])
        action_gradients.append(action_gradient[0][0])
    actions = np.array(batch)[:,2][0]
    states = np.array(batch)[:,3][0]
    critic.fit(sess, states, actions, Q_targets)
    actor.fit(sess, states, actions, action_gradients)
        
    
def train():
    with tf.Session() as sess:
        episodes = 1000
        batch_size = 32
        discount_factor = 0.99

        env = gym.make('MountainCarContinuous-v0')
        action_dim = len(env.action_space.sample())
        state_dim = len(env.reset())
        action_max = env.action_space.high[0]

        memory = deque(maxlen=2000)
        actor = Actor(state_dim, action_dim, action_max, batch_size)
        critic = Critic(state_dim, action_dim, action_max)
        action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))
        sess.run(tf.global_variables_initializer())

        for episode in range(episodes):
            done = False
            state = np.reshape(env.reset(), [1,state_dim])
            total_reward = 0
            while(not done):
                action = actor.predict(sess, np.reshape(state, (1, state_dim))) + action_noise()
                next_state, reward, done, info = env.step(action)
                next_state = np.reshape(next_state,[1,state_dim])
                memory.append([next_state,reward, action, state, done])
                total_reward += reward
                env.render()
                if len(memory) > 32:
                    batch = random.sample(memory,batch_size)
                    fit_batch(sess, batch, actor, critic, discount_factor)
                    actor.update_net(sess)
                    critic.update_net(sess)
                    
            print('episode: %d ... total reward: %d' %(episode,total_reward))
                    
train()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
