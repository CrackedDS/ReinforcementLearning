import gym
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v0')

max_steps = 99
episodes = 2000
df = 0.99
epsilon = 0.1
step_list = []
r_list = []

#  NetworkDesign():
tf.reset_default_graph()
q_network = tf.Graph()
with q_network.as_default():
    inputs = tf.placeholder(shape=[1,16], dtype=tf.float32)
    weights = tf.Variable(tf.random_uniform(shape=[16,4], minval=0, maxval=0.1))
    q_pred = tf.matmul(inputs, weights)
    predict = tf.argmax(q_pred, axis=1)

    q_actual = tf.placeholder(shape=[1,4], dtype=tf.float32)
    loss = tf.reduce_sum(tf.square(q_actual - q_pred))
    trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    update_model = trainer.minimize(loss)
    init = tf.global_variables_initializer() # initialize all the graph variables

# NetworkTrain(q_network):
with tf.Session(graph=q_network) as sess:
    sess.run(init)
    for i in range(episodes):
        s = env.reset()
        rAll = 0
        episode_step = 0
        d = False

        while episode_step < max_steps:
            episode_step += 1

            action, q_values_s = sess.run([predict, q_pred], feed_dict={inputs:np.identity(16)[s:s+1]})
            if np.random.rand(1) < epsilon:
                action[0] = env.action_space.sample()

            s1, r, d, _ = env.step(action[0])

            q_values_s1 = sess.run([q_pred], feed_dict={inputs:np.identity(16)[s1:s1+1]})
            max_q_s1 = np.max(q_values_s1)

            q_actual_s = q_values_s
            q_actual_s[0, action[0]] = r + df * max_q_s1
            _, weights_updated = sess.run([update_model, weights], feed_dict={inputs:np.identity(16)[s:s+1], q_actual:q_actual_s})

            rAll += r
            s = s1

            if d == True:
                epsilon = 1. / ((i / 50) + 10)
                break

        step_list.append(episode_step)
        r_list.append(rAll)

print("Percentage successful episodes:", sum(r_list)/episodes, "%")
