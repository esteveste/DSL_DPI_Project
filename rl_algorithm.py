import gym
import numpy as np

from stable_baselines3 import PPO, A2C, DQN

env = gym.make('CartPole-v0')

model = DQN('MlpPolicy', env, verbose=0)
model.learn(total_timesteps=100000)



runs = 100

total_scores = np.zeros(runs)

for i in range(runs):
    done = False
    obs = env.reset()
    ep_reward = 0

    #if render: env.render()

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, r, done, _ = env.step(action)
        ep_reward += r

        #if render: env.render()
    total_scores[i] = ep_reward

print(total_scores.mean())
print(total_scores.std())
env.close()