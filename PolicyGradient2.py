'''
Multi-armed Bandit problem with more than one bandit.
States present since many bandits.
State -> Action -> Reward example
Action taken doesn't affect the state of the environment
'''
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

class Bandits():
    def __init__(self, num_bandits, num_arms):
        self.state = 0 # Which bandit are you picking
        self.num_arms = num_arms
        self.num_bandits = num_bandits
        self.bandits = 10 * np.random.random_sample((num_bandits, num_arms)) - 5
        # self.bandits = np.array([[0.2,0,-0.0,-5],[0.1,-5,1,0.25],[-5,5,5,5]]) # This combo makes agent learn correctly for the designed network and hyperparameters

    def pullArm(self, arm):
        rand_num = np.random.randn(1)
        if rand_num < self.bandits[self.state, arm]:
            return -1
        else:
            return 1

    def getBandit(self):
        self.state = np.random.randint(0, self.num_bandits) # State chosen for environment not dependent on action
        return self.state

class Agent():
    def __init__(self, lr, state_size, action_size):
        self.state_chosen = tf.placeholder(shape=[1], dtype=tf.int32)
        state_one_hot = slim.one_hot_encoding(self.state_chosen, state_size)
        output = slim.fully_connected(state_one_hot, action_size, biases_initializer=None, activation_fn=tf.nn.sigmoid, weights_initializer=tf.ones_initializer())
        self.output = tf.reshape(output, [-1])
        self.chosen_action = tf.argmax(self.output, 0)

        self.reward_holder = tf.placeholder(shape=[1], dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[1], dtype=tf.int32)
        self.responsible_weight = tf.slice(self.output, self.action_holder, [1])
        self.loss = -(tf.log(self.responsible_weight) * self.reward_holder)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
        self.update = optimizer.minimize(self.loss)

tf.reset_default_graph()
bandits = Bandits(3,4)
agent = Agent(lr=0.001, state_size=bandits.num_bandits, action_size=bandits.num_arms)
total_episodes = 10000
total_reward = np.zeros(shape=[bandits.num_bandits, bandits.num_arms])
epsilon = 0.1
weights = tf.trainable_variables()[0]
init = tf.global_variables_initializer()

print("Bandits:")
for i in range(bandits.num_bandits):
    print("Bandit", i+1, bandits.bandits[i])

with tf.Session() as sess:
    sess.run(init)
    for episode in range(total_episodes):
        chosen_bandit = bandits.getBandit()
        if np.random.rand(1) < epsilon:
            chosen_action = np.random.randint(bandits.num_arms)
        else:
            chosen_action = sess.run(agent.chosen_action, feed_dict={agent.state_chosen:[chosen_bandit]})
        reward = bandits.pullArm(chosen_action)
        _, ww = sess.run([agent.update, weights], feed_dict={agent.state_chosen:[chosen_bandit], agent.reward_holder:[reward], agent.action_holder:[chosen_action]})
        total_reward[chosen_bandit, chosen_action] += reward
        if episode % 500 == 0:
            print("Mean reward for the", bandits.num_bandits, " bandits are:", np.mean(total_reward, axis=1))

for i in range(bandits.num_bandits):
    print("The agent thinks the action", np.argmax(ww[i])+1, "is best for bandit", i+1)
    if np.argmin(bandits.bandits[i]) == np.argmax(ww[i]):
        print("...and it is right!")
    else:
        print("...and it is wrong!")
