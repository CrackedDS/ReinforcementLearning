'''
Implementation of DQN with Experience Buffer, separate Target Q-Network, Double DQN, Dueling DQN
'''

import tensorflow as tf
import gym
import tensorflow.contrib.slim as slim
import numpy as np
import os
import random

from gridworld import gameEnv
env = gameEnv(partial=False, size=5)

# The path to save our model to.
path = "./DeepMemory"
if not os.path.exists(path):
    os.makedirs(path)

# Experience Replay
class experienceBuffer():
    def __init__(self, buffer_size=50000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:len(experience)+len(self.buffer)-self.buffer_size] = []
        self.buffer.extend(experience)

    def sample(self, size):
        return np.reshape(np.array(random.sample(self.buffer, size)), [size, 5])

# Q Network Parameters
q_input_size = 21168
q_hidden1_size = 32
q_hidden1_kernel = [8,8]
q_hidden1_stride = [4,4]
q_hidden1 = [q_hidden1_size, q_hidden1_kernel, q_hidden1_stride]
q_hidden2_size = 64
q_hidden2_kernel = [4,4]
q_hidden2_stride = [2,2]
q_hidden2 = [q_hidden2_size, q_hidden2_kernel, q_hidden2_stride]
q_hidden3_size = 64
q_hidden3_kernel = [3,3]
q_hidden3_stride = [1,1]
q_hidden3 = [q_hidden3_size, q_hidden3_kernel, q_hidden3_stride]
q_hidden4_size = 512
q_hidden4_kernel = [7,7]
q_hidden4_stride = [1,1]
q_hidden4 = [q_hidden4_size, q_hidden4_kernel, q_hidden4_stride]

class QNetwork():
    def __init__(self, lr, input_size, hidden1, hidden2, hidden3, hidden4):
        # Input image initial manipulation
        self.input = tf.placeholder(shape=[None, input_size], dtype=tf.float32)
        self.input_image = tf.reshape(self.input, shape=[-1,int(np.sqrt(input_size/3)),int(np.sqrt(input_size/3)),3])
        self.conv1 = slim.conv2d(inputs=self.input_image,num_outputs=hidden1[0],kernel_size=hidden1[1],stride=hidden1[2],padding='VALID', biases_initializer=None)
        self.conv2 = slim.conv2d(inputs=self.conv1,num_outputs=hidden2[0],kernel_size=hidden2[1],stride=hidden2[2],padding='VALID', biases_initializer=None)
        self.conv3 = slim.conv2d(inputs=self.conv2,num_outputs=hidden3[0],kernel_size=hidden3[1],stride=hidden3[2],padding='VALID', biases_initializer=None)
        self.conv4 = slim.conv2d(inputs=self.conv3,num_outputs=hidden4[0],kernel_size=hidden4[1],stride=hidden4[2],padding='VALID', biases_initializer=None)

        # Dueling DQN
        # Stage 1: Finding the Value and Advantage
        self.advantage_in_stream, self.value_in_stream = tf.split(self.conv4, 2, 3)
        self.advantage_in = slim.flatten(self.advantage_in_stream)
        self.value_in = slim.flatten(self.value_in_stream)
        xavier_init = tf.contrib.layers.xavier_initializer()
        self.adv_weights = tf.Variable(xavier_init([hidden4[0]//2, env.actions]))
        self.val_weights = tf.Variable(xavier_init([hidden4[0]//2, 1]))
        self.advantage = tf.matmul(self.advantage_in, self.adv_weights)
        self.value = tf.matmul(self.value_in, self.val_weights)

        # Stage 2: Getting Q-Values from values and advatnages
        self.q_values = self.value + tf.subtract(self.advantage, tf.reduce_mean(self.advantage, axis=1, keepdims=True))
        self.predict_action = tf.argmax(self.q_values, axis=1)

        # Double DQN (setup)
        self.target_q = tf.placeholder(shape=[None], dtype=tf.float32)
        self.action = tf.placeholder(shape=[None], dtype=tf.int32)
        self.action_onehot = tf.one_hot(self.action, env.actions, dtype=tf.float32)

        self.Q = tf.reduce_sum(tf.multiply(self.q_values, self.action_onehot), axis=1)

        self.loss = tf.reduce_mean(tf.square(self.target_q - self.Q))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.update = self.optimizer.minimize(self.loss)

class Agent():
    def __init__(self, lr, q_input_size, q_hidden1, q_hidden2, q_hidden3, q_hidden4):
        self.mainQN = QNetwork(lr, q_input_size, q_hidden1, q_hidden2, q_hidden3, q_hidden4)
        self.targetQN = QNetwork(lr, q_input_size, q_hidden1, q_hidden2, q_hidden3, q_hidden4)
        self.training_vars = tf.trainable_variables()
        self.target_ops = self.updateTargetGraph(self.training_vars, tau)
        self.epsilon = start_epsilon
        self.step_drop = (start_epsilon - end_epsilon) / annealing_steps
        self.step_list = []
        self.reward_list = []
        self.lifetime_steps = 0
        self.experience_buffer = experienceBuffer()

    def qInputShaper(self, state):
        return np.reshape(state, [q_input_size])

    def updateTargetGraph(self, tfVars, tau):
        total_vars = len(tfVars)
        op_holder = []
        for idx,var in enumerate(tfVars[0:total_vars//2]):
            op_holder.append(tfVars[idx+total_vars//2].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_vars//2].value())))
        return op_holder

    def updateTarget(self, sess, op_holder):
        for op in op_holder:
            sess.run(op)

# Training parameters
lr = 1e-2 # learning rate
df = 0.99 # discount factor
batch_size = 32 # Batch processing of the experience buffer
update_freq = 4 # Frequency to update the Agent Q-Network
start_epsilon = 1 #Starting chance of random action
end_epsilon = 0.1 #Final chance of random action
annealing_steps = 10000. #How many steps of training to reduce start_epsilon to end_epsilon.
episodes = 10000 #How many episodes of game environment to train network with.
pre_train_steps = 10000 #How many steps of random actions before training begins.
max_steps = 50 #The max allowed length of our episode.
load_model = False #Whether to load a saved model.
tau = 0.001 #Rate to update target network toward primary network
checkpoint = 1000 # Checkpoint to save the model


tf.reset_default_graph()
agent = Agent(lr, q_input_size, q_hidden1, q_hidden2, q_hidden3, q_hidden4)
init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    if load_model == True:
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(path)
        saver.restore(sess, ckpt.model_checkpoint_path)

    for episode in range(episodes):
        episode_buffer = experienceBuffer()
        episode_reward = 0
        s = env.reset()
        s = agent.qInputShaper(s)
        episode_step = 0

        while episode_step < max_steps:
            # epsilon greedy policy
            if np.random.rand(1) < agent.epsilon or agent.lifetime_steps < pre_train_steps:
                a = np.random.randint(0,4)
            else:
                a = sess.run(agent.mainQN.predict_action, feed_dict={agent.mainQN.input:[s]})

            s1, r, d = env.step(a)
            s1 = agent.qInputShaper(s1)
            agent.lifetime_steps += 1
            episode_step += 1
            episode_buffer.add(np.reshape(np.array([s,a,r,s1,d]), [1,5]))

            if agent.lifetime_steps > pre_train_steps:
                # epsilon decay
                if agent.epsilon > end_epsilon:
                    agent.epsilon -= agent.step_drop

                # training if every update freq after pre training steps
                if agent.lifetime_steps % update_freq == 0:
                    training_batch = agent.experience_buffer.sample(batch_size)
                    # Double DQN
                    estimate_action = sess.run(agent.mainQN.predict_action, feed_dict={agent.mainQN.input:np.vstack(training_batch[:,3])})
                    target_qs_s1 = sess.run(agent.targetQN.q_values, feed_dict={agent.targetQN.input:np.vstack(training_batch[:,3])})
                    target_q_s1 = target_qs_s1[range(batch_size), estimate_action]
                    target_q = training_batch[:,2] + (df * target_q_s1 * (1-training_batch[:,4]))

                    # Update main Q Network
                    _ = sess.run(agent.mainQN.update, feed_dict={agent.mainQN.input:np.vstack(training_batch[:,0]), agent.mainQN.target_q:target_q, agent.mainQN.action:training_batch[:,1]})
                    # Update target Q Network
                    agent.updateTarget(sess, agent.target_ops)
            episode_reward += r
            s = s1
            if d == True:
                break
        agent.experience_buffer.add(episode_buffer.buffer)
        agent.step_list.append(episode_step)
        agent.reward_list.append(episode_reward)

        # Checkpoint for the model
        if episode % checkpoint == 0:
            saver.save(sess, path + '/model-' + str(episode) + '.ckpt')
            print("Saved Model at episode", episode)
        if len(agent.reward_list) % 10 == 0:
            print("LifeTime Steps:", agent.lifetime_steps, "Mean Reward of last 10 episodes:", np.mean(agent.reward_list[-10:]), "Epsilon:", agent.epsilon)
    saver.save(sess, path + '/model-' + str(episode) + '.ckpt')
print("Percent of succesful episodes:", sum(agent.reward_list) / episodes, "%")
