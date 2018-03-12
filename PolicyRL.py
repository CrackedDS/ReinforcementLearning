'''
Cartpole problem.
States present as position of the pole.
State -> Action -> (State, Reward) example
Action taken affects the state of the environment.
'''

import tensorflow as tf
import numpy as np
import gym
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt

env = gym.make('CartPole-v0')
df = 0.99

def discountedReward(r):
    discount_reward = np.zeros_like(r)
    prev_disc_reward = 0
    for i in reversed(range(len(r))):
        discount_reward[i] = df * prev_disc_reward + r[i]
        prev_disc_reward = discount_reward[i]
    return discount_reward

class Agent():
    def __init__(self, lr, s_size, a_size, h_size):
        self.state_input = tf.placeholder(shape=[None,s_size], dtype=tf.float32)
        hidden = slim.fully_connected(self.state_input, h_size, activation_fn=tf.nn.relu, biases_initializer=None)
        self.output = slim.fully_connected(hidden, a_size, activation_fn=tf.nn.softmax, biases_initializer=None)

        self.action_holder = tf.placeholder(shape=[None], dtype=tf.int32)
        self.reward_holder = tf.placeholder(shape=[None], dtype=tf.float32)

        self.action_indexes = tf.range(0, tf.shape(self.output)[0]) * tf.shape(self.output)[1] + self.action_holder
        self.responsible_actions = tf.gather(tf.reshape(self.output, [-1]), self.action_indexes)
        self.loss = -tf.reduce_mean(tf.log(self.responsible_actions) * self.reward_holder)
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.training_vars = tf.trainable_variables()
        self.gradients = optimizer.compute_gradients(self.loss, self.training_vars)
        self.gradients_holder = [tf.placeholder(shape=np.shape(i), dtype=tf.float32) for i in self.training_vars]
        self.update = optimizer.apply_gradients(zip(self.gradients_holder, self.training_vars))

tf.reset_default_graph()
agent = Agent(lr=1e-2,s_size=4,a_size=2,h_size=8)
episodes = 5000
max_steps = 999
update_freq = 5
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    total_reward = []
    total_steps = []
    grad_cumulative = sess.run(agent.training_vars)
    grad_cumulative = [0 for _ in grad_cumulative]
    
    for episode in range(episodes):
        s = env.reset()
        episode_reward = 0
        experience_buffer = []
        for step in range(max_steps):
            net_action = sess.run(agent.output, feed_dict={agent.state_input:[s]})
            picked_action = np.random.choice(net_action[0], p=net_action[0])
            a = np.argmax(net_action == picked_action)

            s1, r, d, _ = env.step(a)
            experience_buffer.append([s,a,r,s1])
            s = s1
            episode_reward += r
            if d == True:
                experience_buffer = np.array(experience_buffer)
                experience_buffer[:, 2] = discountedReward(experience_buffer[:, 2])
                grads = sess.run(agent.gradients, feed_dict={agent.state_input:np.vstack(experience_buffer[:,0]), agent.action_holder:experience_buffer[:,1], agent.reward_holder:experience_buffer[:,2]})
                grads = [g for (g,v) in grads]
                grad_cumulative = [x+y for x,y in zip(grad_cumulative, grads)]
                if episode % update_freq == 0 and episode != 0:
                    _ = sess.run(agent.update, feed_dict=dict(zip(agent.gradients_holder, grad_cumulative)))
                    grad_cumulative = [0 for _ in grad_cumulative]
                total_reward.append(episode_reward)
                total_steps.append(step)
                break
        if episode % 100 == 0:
            print("Reward:", np.mean(total_reward[-100:]), "Steps:", np.mean(total_steps[-100:]))
