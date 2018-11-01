
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
        self.input_state_pred, _ , self.output_pred , _ = self.create_net('pred_net_actor')
        self.weights = tf.trainable_variables('actor')
        self.weights_pred = tf.trainable_variables('pred_net_actor')

        self.update_pred_net = [ tf.assign(self.weights_pred[i], 
                                        tf.add(  tf.multiply(self.tau, self.weights[i]),
                                                 tf.multiply(1.0-self.tau, self.weights_pred[i]) ))
                                for i in range(len(self.weights))]
        
        self.action_gradient = tf.placeholder('float', shape=[None,self.action_dim])
        self.J_gradient = tf.gradients(self.output, self.weights, -self.action_gradient)
        self.J_gradient_normalized = list(map(lambda x: x/batch_size, self.J_gradient))
        self.optimizer = AdamOptimizer(learning_rate=self.lr).apply_gradients(zip(self.J_gradient_normalized, self.weights))
        
        
   
    def predict(self, sess, state):
        return sess.run(self.output_pred, feed_dict = {self.input_state_pred : state} )
        
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
            n1 = fully_connected(inputs=inputs_state, num_outputs=400, activation_fn=relu)
            n1_normed = tf.layers.batch_normalization(n1)
            
            n2 = fully_connected(inputs=n1_normed, num_outputs=300, activation_fn=relu)
            n2_normed = tf.layers.batch_normalization(n2)
            
            w_init = tf.initializers.random_uniform(minval=-0.003, maxval=0.003)
            output = fully_connected(n2_normed, self.action_dim, activation_fn=tanh, weights_initializer=w_init)
            output_scaled = tf.multiply(output, self.action_max_value)
            
            loss = mean_squared_error(targets, output_scaled)
    
        return inputs_state, targets, output_scaled, loss        
    
    
class Critic():
    def __init__(self, state_size, action_size, action_max_value):
        self.gamma = 0.95
        self.tau = 0.001
        self.lr = 0.001
        self.state_dim = state_size
        self.action_dim = action_size
        self.action_max_value = action_max_value
        
        self.input_state, self.input_action, self.targets, self.output, self.loss = self.create_net('critic')
        self.input_state_pred, self.input_action_pred , _ , self.output_pred , _ = self.create_net('pred_net_critic')
        self.weights = tf.trainable_variables('critic')
        self.weights_pred = tf.trainable_variables('pred_net_critic')
            
        self.update_pred_net = [ tf.assign(self.weights_pred[i], 
                                        tf.add(  tf.multiply(self.tau,self.weights[i]),
                                                 tf.multiply(1.0-self.tau, self.weights_pred[i]) ))
                                for i in range(len(self.weights))]
        self.optimizer = AdamOptimizer().minimize(self.loss)
        self.action_gradient = tf.gradients(self.loss, self.input_action)
   
    def predict(self, sess, state, action):
        return sess.run(self.output_pred, feed_dict = { self.input_state_pred : state,
                                                        self.input_action_pred : action})
        
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
            n1 = fully_connected(inputs=inputs_state, num_outputs=400, activation_fn=relu, scope='l1')
            n1_normed = tf.layers.batch_normalization(n1)
            n2 = fully_connected(inputs=n1_normed, num_outputs=300, activation_fn=relu, scope='l2')
            n2_normed = tf.layers.batch_normalization(n2)
            actions = fully_connected(inputs=inputs_action, num_outputs=300, activation_fn=relu, scope='action_layer')
            
            n2_added = tf.add(actions,n2_normed)
            
            output_unscaled = fully_connected(inputs=n2_added, num_outputs=self.action_dim, activation_fn=relu, scope='n3')
           
            w_init = tf.initializers.random_uniform(minval=-0.003, maxval=0.003)
            output = fully_connected(inputs = output_unscaled, num_outputs=1, weights_initializer=w_init)
            
            loss = mean_squared_error(targets, output)
        
         
        return inputs_state, inputs_action, targets, output, loss 
    
def fit_batch(sess, batch, actor, critic, discount):
    Q_targets = []
    states = batch[3].tolist()
    actions = batch[2].tolist()
    next_states = batch[0].tolist()
    Q_max_predictions = critic.predict(sess, next_states, actor.predict(sess, states))
    for i in range(len(batch)-1):
        done = batch[4][i]
        reward = batch[1][i]
        Q_target = reward 
        if(not done):
            Q_target += discount*Q_max_predictions[i][0]
        Q_targets.append(Q_target)  
    action_gradients = critic.get_action_gradient(sess, states, actions, Q_targets)
    critic.fit(sess, states, actions, Q_targets)
    actor.fit(sess, states, actions, action_gradients[0])

def get_batch(memory, batch_size):
    batch = random.sample(memory,batch_size)
    return np.array(batch).T
    
def train():
    with tf.Session() as sess:
        episodes = 1000
        batch_size = 4
        discount_factor = 0.99

        env = gym.make('Pendulum-v0')
        action_dim = len(env.action_space.sample())
        state_dim = len(env.reset())
        action_max = env.action_space.high[0]

        memory = deque(maxlen=20000)
        actor = Actor(state_dim, action_dim, action_max, batch_size)
        critic = Critic(state_dim, action_dim, action_max)
        action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))
        sess.run(tf.global_variables_initializer())

        for episode in range(episodes):
            done = False
            state = env.reset()
            total_reward = 0
            while(not done):
                action = actor.predict(sess, np.reshape(state, (1, state_dim))) + action_noise()
                next_state, reward, done, info = env.step(action)
                memory.append([np.squeeze(next_state),reward, action[0], np.squeeze(state), done])
                total_reward += reward
                env.render()
                if len(memory) > batch_size:
                    batch = get_batch(memory, batch_size)
                    fit_batch(sess, list(batch), actor, critic, discount_factor)
                    actor.update_net(sess)
                    critic.update_net(sess)
                state = next_state    
            print('episode: %d ... total reward: %d' %(episode,total_reward))
                    
train()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
