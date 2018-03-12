import numpy as np
import tensorflow as tf
import gym

real_env = gym.make("CartPole-v0")

lr = 1e-2
df = 0.99

policy_hidden_s = 8
policy_input_s = 4
policy_output_s = 1
venv_input_s = 5
venv_hidden1_s = 256
venv_hidden2_s = 256
venv_output_s = [4,1,1]

model_bs = 3
real_bs = 3

def discountedRewards(r):
    discounted_reward = np.zeros_like(r)
    discounted_reward[len(r)-1] = r[len(r)-1]
    for i in reversed(range(len(r)-1)):
        discounted_reward[i] = r[i] + df * discounted_reward[i+1]
    return discounted_reward

class Agent():
    def __init__(self, lr, input_s, hidden_s, output_s):
        self.state_input = tf.placeholder(shape=[None, input_s], dtype=tf.float32)
        self.__AW1 = tf.get_variable("AW1", shape=[input_s, hidden_s], initializer=tf.contrib.layers.xavier_initializer())
        self.hidden_layer = tf.nn.relu(tf.matmul(self.state_input, self.__AW1))
        self.__AW2 = tf.get_variable("AW2", shape=[hidden_s, output_s], initializer=tf.contrib.layers.xavier_initializer())
        self.output_layer = tf.nn.sigmoid(tf.matmul(self.hidden_layer, self.__AW2))

        self.action_holder = tf.placeholder(shape=[None], dtype=tf.float32)
        self.reward_holder = tf.placeholder(shape=[None], dtype=tf.float32)

        self.loss = -tf.reduce_mean(tf.log((self.action_holder * self.output_layer) + ((1 - self.action_holder) * (1 - self.output_layer))) * self.reward_holder)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.training_vars = tf.trainable_variables()
        self.gradients = self.optimizer.compute_gradients(self.loss, self.training_vars)
        self.gradient_holder = [tf.placeholder(shape=np.shape(j), dtype=tf.float32) for j in self.training_vars]
        self.update = self.optimizer.apply_gradients(zip(self.gradient_holder, self.training_vars))

class Virtual_Env():
    def __init__(self, lr, input_s, hidden1_s, hidden2_s, output_s):
        self.state_input = tf.placeholder(shape=[None, input_s], dtype=tf.float32)

        self.__VW1 = tf.get_variable("W1", shape=[input_s, hidden1_s], initializer=tf.contrib.layers.xavier_initializer())
        self.__B1 = tf.Variable(tf.zeros([hidden1_s]), name="B1")
        self.hidden_layer1 = tf.nn.relu(tf.matmul(self.state_input, self.__VW1) + self.__B1)

        self.__VW2 = tf.get_variable("W2", shape=[hidden1_s, hidden2_s], initializer=tf.contrib.layers.xavier_initializer())
        self.__B2 = tf.Variable(tf.zeros([hidden2_s]), name="B2")
        self.hidden_layer2 = tf.nn.relu(tf.matmul(self.hidden_layer1, self.__VW2) + self.__B2)

        self.__Ws = tf.get_variable("Ws", shape=[hidden2_s, output_s[0]], initializer=tf.contrib.layers.xavier_initializer())
        self.__Wr = tf.get_variable("Wr", shape=[hidden2_s, output_s[1]], initializer=tf.contrib.layers.xavier_initializer())
        self.__Wd = tf.get_variable("Wd", shape=[hidden2_s, output_s[2]], initializer=tf.contrib.layers.xavier_initializer())
        self.__Bs = tf.Variable(tf.zeros([output_s[0]]), name="Bs")
        self.__Br = tf.Variable(tf.zeros([output_s[1]]), name="Br")
        self.__Bd = tf.Variable(tf.zeros([output_s[2]]), name="Bd")
        self.new_state = tf.matmul(self.hidden_layer2, self.__Ws) + self.__Bs
        self.reward = tf.matmul(self.hidden_layer2, self.__Wr) + self.__Br
        self.done = tf.sigmoid(tf.matmul(self.hidden_layer2, self.__Wd) + self.__Bd)

        self.new_state_holder = tf.placeholder(shape=[None, output_s[0]], dtype=tf.float32)
        self.reward_holder = tf.placeholder(shape=[None, output_s[1]], dtype=tf.float32)
        self.done_holder = tf.placeholder(shape=[None, output_s[2]], dtype=tf.float32)

        self.state_loss = tf.square(self.new_state_holder - self.new_state)
        self.reward_loss = tf.square(self.reward_holder - self.reward)
        self.done_loss = -tf.log(tf.multiply(self.done, self.done_holder) + tf.multiply((1-self.done), (1-self.done_holder)))
        self.total_loss = tf.reduce_mean(self.state_loss + self.reward_loss + self.done_loss)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.update = self.optimizer.minimize(self.total_loss)

    def step(self, sess, inputs):
        s1, r, d = sess.run([self.new_state, self.reward, self.done], feed_dict={self.state_input:[inputs]})
        d = np.clip(d, 0, 1)
        if d > 0.1:
            d = True
        else:
            d = False
        return s1, r, d

tf.reset_default_graph()
agent = Agent(lr, policy_input_s, policy_hidden_s, policy_output_s)
virtual_env = Virtual_Env(lr, venv_input_s, venv_hidden1_s, venv_hidden2_s, venv_output_s)
episodes = 5000
max_steps = 999
init = tf.global_variables_initializer()

draw_model = False
train_model = True
train_policy = False
switch_episode = 0

with tf.Session() as sess:
    sess.run(init)
    real_total_reward = []
    real_total_steps = []
    model_total_reward = []
    model_total_steps = []
    grad_cumulative = sess.run(agent.training_vars)
    grad_cumulative = [0 for _ in grad_cumulative]

    for episode in range(episodes):
        experience_buffer = []
        episode_reward = 0
        s = None
        if draw_model == False:
            s = real_env.reset()
            batch_size = real_bs
        else:
            s = np.random.uniform(-0.1, 0.1, [4])
            batch_size = model_bs

        for step in range(max_steps):
            if draw_model == False:
                real_env.render()
            a = sess.run(agent.output_layer, feed_dict={agent.state_input:[s]})
            a = 1 if np.random.uniform() < a else 0

            if draw_model == False:
                s1, r, d, _ = real_env.step(a)
            else:
                s1, r, d = virtual_env.step(sess, np.hstack([s, np.array(a)]))

            s1 = np.squeeze(s1)
            a = np.squeeze(a)
            r = np.squeeze(r)

            episode_reward += r
            experience_buffer.append([s, np.array(a), np.array(r), s1, np.array(d*1)])
            s = s1

            if d == True:
                if draw_model == False:
                    real_total_reward.append(episode_reward)
                    real_total_steps.append(step)
                else:
                    model_total_reward.append(episode_reward)
                    model_total_steps.append(step)

                if train_model == True:
                    experience_buffer = np.array(experience_buffer)
                    state_in = np.array([np.hstack([i,j]) for i,j in zip(experience_buffer[:,0], experience_buffer[:,1])])
                    feed_dict = {virtual_env.state_input:state_in,
                                virtual_env.new_state_holder:np.vstack(experience_buffer[:,3]),
                                virtual_env.reward_holder:np.vstack(experience_buffer[:,2]),
                                virtual_env.done_holder:np.vstack(experience_buffer[:,4])}
                    _ = sess.run(virtual_env.update, feed_dict=feed_dict)

                if train_policy == True:
                    experience_buffer = np.array(experience_buffer)
                    discount_reward = discountedRewards(experience_buffer[:,2])
                    grads = sess.run(agent.gradients, feed_dict={agent.state_input:np.vstack(experience_buffer[:,0]),
                                                                agent.action_holder:experience_buffer[:,1],
                                                                agent.reward_holder:discount_reward})
                    grads = [g for g,v in grads]
                    grad_cumulative = [x+y for x,y in zip(grads, grad_cumulative)]

                if switch_episode + batch_size == episode:
                    switch_episode = episode

                    if train_policy == True:
                        _ = sess.run(agent.update, feed_dict=dict(zip(agent.gradient_holder, grad_cumulative)))
                        grad_cumulative = [0 for _ in grad_cumulative]

                    if draw_model == False:
                        print("Real Reward:", np.mean(real_total_reward[-3:]), "Steps:", np.mean(real_total_steps[-3:]))
                    else:
                        print("Model Reward:", np.mean(model_total_reward[-3:]), "Steps:", np.mean(model_total_steps[-3:]))

                    if episode > 100:
                        draw_model = not draw_model
                        train_model = not train_model
                        train_policy = not train_policy
                break
