'''
Multi-armed Bandit problem with just one bandit.
No states present since just one bandit.
Action -> Reward example
No state involved.
'''

import tensorflow as tf
import numpy as np

bandits = [3, 0.2, -0.3, -6]
num_bandits = len(bandits)

def drawBandit(bandit):
    random = np.random.randn(1)
    if random < bandit:
        return -1
    else:
        return 1

tf.reset_default_graph()
policy_graph = tf.Graph()
with policy_graph.as_default():
    # inputs = tf.placeholder(shape=[1,4], dtype=tf.float32)
    weights = tf.Variable(tf.ones([num_bandits]))
    chosen_action = tf.argmax(weights,0)

    reward_holder = tf.placeholder(shape=[1], dtype=tf.float32)
    action_holder = tf.placeholder(shape=[1], dtype=tf.int32)
    responsible_weight = tf.slice(weights, action_holder, [1])
    loss = -(tf.log(responsible_weight) * reward_holder)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    update = optimizer.minimize(loss)
    init = tf.global_variables_initializer()

total_episodes = 1000 #Set total number of episodes to train agent on.
total_reward = np.zeros(num_bandits) #Set scoreboard for bandits to 0.
e = 0.1 #Set the chance of taking a random action.

# Launch the tensorflow graph
with tf.Session(graph=policy_graph) as sess:
    sess.run(init)
    for i in range(total_episodes):
        #Choose either a random action or one from our network.
        if np.random.rand(1) < e:
            action = np.random.randint(num_bandits)
        else:
            action = sess.run(chosen_action)

        reward = drawBandit(bandits[action]) #Get our reward from picking one of the bandits.

        #Update the network.
        _,resp,ww = sess.run([update,responsible_weight,weights], feed_dict={reward_holder:[reward],action_holder:[action]})

        #Update our running tally of scores.
        total_reward[action] += reward
        if i % 50 == 0:
            print("Running reward for the", num_bandits, "bandits:", total_reward)

print("The agent thinks bandit", np.argmax(ww)+1, "is the most promising....")
if np.argmax(ww) == np.argmax(-np.array(bandits)):
    print("...and it was right!")
else:
    print("...and it was wrong!")
