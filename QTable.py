import gym
import numpy as np

env = gym.make('FrozenLake-v0')

QTable = np.zeros([env.observation_space.n, env.action_space.n])

lr = 0.8
df = .95
episodes = 2000
rList = []

for i in range(episodes):
    s = env.reset()
    rAll = 0
    d = False
    j = 0
    while j < 99:
        j += 1
        a = np.argmax(QTable[s,:] + np.random.randn(1, env.action_space.n) * (1./(i+1)))
        s1, r, d, _ = env.step(a)
        QTable[s, a] += lr * (r + df * np.max(QTable[s1, :]) - QTable[s, a])
        rAll += r
        s =  s1
        if d == True:
            break
    rList.append(rAll)

print("Score over time:", str(sum(rList)/episodes))
print("Final Q Table\n", QTable)
