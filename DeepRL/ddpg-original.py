""" 
Implementation of DDPG - Deep Deterministic Policy Gradient
Algorithm and hyperparameter details can be found here: 
    http://arxiv.org/pdf/1509.02971v2.pdf
The algorithm is tested on the Pendulum-v0 OpenAI gym task 
and developed with tflearn + Tensorflow
Author: Patrick Emami
"""
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected 
from tensorflow.nn import relu, tanh
import numpy as np
import gym
from gym import wrappers
import tflearn
import argparse
import pprint as pp

from collections import deque
import random

class ReplayBuffer(object):

    def __init__(self, buffer_size, random_seed=123):
        """
        The right side of the deque contains the most recent experiences 
        """
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        random.seed(random_seed)

    def add(self, s, a, r, t, s2):
        experience = (s, a, r, t, s2)
        if self.count < self.buffer_size: 
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        batch = []

        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch])
        t_batch = np.array([_[3] for _ in batch])
        s2_batch = np.array([_[4] for _ in batch])

        return s_batch, a_batch, r_batch, t_batch, s2_batch

    def clear(self):
        self.buffer.clear()
        self.count = 0


# ===========================
#   Actor and Critic DNNs
# ===========================

class Actor(object):
    """
    Input to the network is the state, output is the action
    under a deterministic policy.
    The output layer activation is a tanh to keep the action
    between -action_bound and action_bound
    """

    def __init__(self, sess, state_dim, action_dim, action_bound,  batch_size):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = 0.0001
        self.tau = 0.001
        self.batch_size = batch_size

        # Actor Network
        self.inputs, self.out, self.scaled_out = self.create_actor_network()

        self.network_params = tf.trainable_variables()

        # Target Network
        self.target_inputs, self.target_out, self.target_scaled_out = self.create_actor_network()

        self.target_network_params = tf.trainable_variables()[
            len(self.network_params):]

        # Op for periodically updating target network with online network
        # weights
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
                                                  tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # This gradient will be provided by the critic network
        self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])

        # Combine the gradients here
        self.unnormalized_actor_gradients = tf.gradients(
            self.scaled_out, self.network_params, -self.action_gradient)
        self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_gradients))

        # Optimization Op
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).\
            apply_gradients(zip(self.actor_gradients, self.network_params))

        self.num_trainable_vars = len(
            self.network_params) + len(self.target_network_params)
       
    def create_actor_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        net = tflearn.fully_connected(inputs, 400)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        net = tflearn.fully_connected(net, 300)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        # Final layer weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(
            net, self.a_dim, activation='tanh', weights_init=w_init)
        # Scale output to -action_bound to action_bound
        scaled_out = tf.multiply(out, self.action_bound)
        return inputs, out, scaled_out

    def train(self, inputs, a_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient
        })

    def predict(self, inputs):
        return self.sess.run(self.scaled_out, feed_dict={
            self.inputs: inputs
        })

    def predict_target(self, inputs):
        return self.sess.run(self.target_scaled_out, feed_dict={
            self.target_inputs: inputs
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars


class Critic(object):
    """
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.
    """

    def __init__(self, sess, state_dim, action_dim):
        self.gamma = 0.95
        self.tau = 0.001
        self.lr = 0.001
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.sess = sess
        
        self.inputs, self.action, self.out = self.create_net('critic')
        self.target_inputs, self.target_action, self.target_out = self.create_net('pred_net_critic')
        self.weights = tf.trainable_variables('critic')
        self.weights_pred = tf.trainable_variables('pred_net_critic')
            
        self.update_target_network_params = [ tf.assign(self.weights_pred[i], 
                                        tf.add(  tf.multiply(self.tau,self.weights[i]),
                                                 tf.multiply(1.0-self.tau, self.weights_pred[i]) ))
                                for i in range(len(self.weights))]
        
        self.predicted_q_value = tf.placeholder(dtype='float', shape=[None,1], name='targets')
        self.loss = tf.losses.mean_squared_error(self.predicted_q_value, self.out)
        
        self.optimize = tf.train.AdamOptimizer().minimize(self.loss)
        self.action_grads = tf.gradients(self.out, self.action)

    def create_net(self, scope_name):
        inputs_state = tf.placeholder(dtype='float', shape=[None,self.s_dim])
        inputs_action = tf.placeholder(dtype='float', shape=[None,self.a_dim])
        
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            n1 = fully_connected(inputs=inputs_state, num_outputs=400, activation_fn=relu )
            n1_normed = tf.layers.batch_normalization(n1)
            n2 = fully_connected(inputs=n1_normed, num_outputs=300, activation_fn=relu )
            n2_normed = tf.layers.batch_normalization(n2)
            actions = fully_connected(inputs=inputs_action, num_outputs=300, activation_fn=relu)
            
            n2_added = tf.add(actions,n2_normed)
            
            output_unscaled = fully_connected(inputs=n2_added, num_outputs=self.a_dim, activation_fn=relu)
           
            w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
            output = tflearn.fully_connected(output_unscaled, 1, weights_init=w_init)
            
        return inputs_state, inputs_action, output

    def train(self, inputs, action, predicted_q_value):
        print(self.sess.run(self.loss, feed_dict={self.inputs: inputs,
                                                  self.action: action,
                                                  self.predicted_q_value: predicted_q_value}))
        return self.sess.run([self.out, self.optimize], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.predicted_q_value: predicted_q_value
        })

    def predict(self, inputs, action):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    def predict_target(self, inputs, action):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action
        })

    def action_gradients(self, inputs, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

# Taken from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py, which is
# based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

# ===========================
#   Tensorflow Summary Ops
# ===========================

def build_summaries():
    episode_reward = tf.Variable(0.)
    tf.summary.scalar("Reward", episode_reward)
    episode_ave_max_q = tf.Variable(0.)
    tf.summary.scalar("Qmax Value", episode_ave_max_q)

    summary_vars = [episode_reward, episode_ave_max_q]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars

# ===========================
#   Agent Training
# ===========================
def fit_batch(sess, batch, actor, critic, discount):
    Q_targets = []
    states = batch[3].tolist()
    actions = batch[2].tolist()
    next_states = batch[0].tolist()
    Q_max_predictions = critic.predict_target( next_states, actor.predict(states))
    for i in range(len(batch[0])):
        done = batch[4][i]
        reward = batch[1][i]
        Q_target = reward 
        if(not done):
            Q_target += discount*Q_max_predictions[i][0]
        Q_targets.append([Q_target]) 
    print(Q_targets)    
    action_gradients = critic.action_gradients( states, actions)
    critic.train( states, actions, Q_targets)
    actor.train( states, action_gradients[0])
    
def get_batch(memory, batch_size):
    batch = random.sample(memory,batch_size)
    return np.array(batch).T

def train():
    with tf.Session() as sess:
        episodes = 1000
        batch_size = 32
        discount_factor = 0.99

        env = gym.make('Pendulum-v0')
        action_dim = len(env.action_space.sample())
        state_dim = len(env.reset())
        action_max = env.action_space.high[0]

        memory = deque(maxlen=20000)
        actor = Actor(sess, state_dim, action_dim, action_max, batch_size)
        critic = Critic(sess, state_dim, action_dim)
        action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))
        sess.run(tf.global_variables_initializer())

        for episode in range(episodes):
            done = False
            state = env.reset()
            total_reward = 0
            while(not done):
                action = actor.predict_target( np.reshape(state, (1, state_dim))) + action_noise()
                next_state, reward, done, info = env.step(action)
                memory.append([np.squeeze(next_state),reward, action[0], np.squeeze(state), done])
                total_reward += reward
                env.render()
                state = next_state 
                if len(memory) > batch_size:
                    batch = get_batch(memory, batch_size)
                    fit_batch(sess, list(batch), actor, critic, discount_factor)
                    actor.update_net(sess)
                    critic.update_net(sess)
                   
            print('episode: %d ... total reward: %d' %(episode,total_reward))
                    
train()
"""
def train(sess, env, args, actor, critic, actor_noise):

    # Set up summary Ops
    summary_ops, summary_vars = build_summaries()

    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(args['summary_dir'], sess.graph)

    # Initialize target network weights
    actor.update_target_network()
    critic.update_target_network()

    # Initialize replay memory
    memory = deque(maxlen=100000)

    # Needed to enable BatchNorm. 
    # This hurts the performance on Pendulum but could be useful
    # in other environments.
    # tflearn.is_training(True)

    for i in range(int(args['max_episodes'])):

        s = env.reset()

        ep_reward = 0
        ep_ave_max_q = 0

        for j in range(int(args['max_episode_len'])):

            if args['render_env']:
                env.render()

            # Added exploration noise
            #a = actor.predict(np.reshape(s, (1, 3))) + (1. / (1. + i))
            a = actor.predict(np.reshape(s, (1, actor.s_dim))) + actor_noise()
            s2, r, terminal, info = env.step(a[0])
            
            memory.append([np.squeeze(s2),r, a[0], np.squeeze(s), terminal])
            

            # Keep adding experience to the memory until
            # there are at least minibatch size samples
            if len(memory) > int(args['minibatch_size']):
                batch = get_batch(memory, 32)
                fit_batch(sess, list(batch), actor, critic, 0.99)

                # Update target networks
                actor.update_target_network()
                critic.update_target_network()

            s = s2
            ep_reward += r

            if terminal:

                summary_str = sess.run(summary_ops, feed_dict={
                    summary_vars[0]: ep_reward,
                    summary_vars[1]: ep_ave_max_q / float(j)
                })

                writer.add_summary(summary_str, i)
                writer.flush()

                print('| Reward: {:d} | Episode: {:d} | Qmax: {:.4f}'.format(int(ep_reward), \
                        i, (ep_ave_max_q / float(j))))
                break

def main(args):

    with tf.Session() as sess:

        env = gym.make(args['env'])
        np.random.seed(int(args['random_seed']))
        tf.set_random_seed(int(args['random_seed']))
        env.seed(int(args['random_seed']))

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        action_bound = env.action_space.high
        # Ensure action bound is symmetric
        assert (env.action_space.high == -env.action_space.low)

        actor = ActorNetwork(sess, state_dim, action_dim, action_bound, 32)

        critic = CriticNetwork(sess, state_dim, action_dim)
        
        actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))

        if args['use_gym_monitor']:
            if not args['render_env']:
                env = wrappers.Monitor(
                    env, args['monitor_dir'], video_callable=False, force=True)
            else:
                env = wrappers.Monitor(env, args['monitor_dir'], force=True)

        train(sess, env, args, actor, critic, actor_noise)

        if args['use_gym_monitor']:
            env.monitor.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='provide arguments for DDPG agent')

    # agent parameters
    parser.add_argument('--actor-lr', help='actor network learning rate', default=0.0001)
    parser.add_argument('--critic-lr', help='critic network learning rate', default=0.001)
    parser.add_argument('--gamma', help='discount factor for critic updates', default=0.99)
    parser.add_argument('--tau', help='soft target update parameter', default=0.001)
    parser.add_argument('--buffer-size', help='max size of the replay buffer', default=1000000)
    parser.add_argument('--minibatch-size', help='size of minibatch for minibatch-SGD', default=64)

    # run parameters
    parser.add_argument('--env', help='choose the gym env- tested on {Pendulum-v0}', default='Pendulum-v0')
    parser.add_argument('--random-seed', help='random seed for repeatability', default=1234)
    parser.add_argument('--max-episodes', help='max num of episodes to do while training', default=50000)
    parser.add_argument('--max-episode-len', help='max length of 1 episode', default=1000)
    parser.add_argument('--render-env', help='render the gym env', action='store_true')
    parser.add_argument('--use-gym-monitor', help='record gym results', action='store_true')
    parser.add_argument('--monitor-dir', help='directory for storing gym results', default='./results/gym_ddpg')
    parser.add_argument('--summary-dir', help='directory for storing tensorboard info', default='./results/tf_ddpg')

    parser.set_defaults(render_env=True)
    parser.set_defaults(use_gym_monitor=False)
    
    args = vars(parser.parse_args())
    
    pp.pprint(args)

    main(args)
"""
