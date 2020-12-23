import gym
import gym_minigrid

import torch
import tianshou
import numpy as np

import random

# gym_minigrid.wrappers.FlatObsWrapper

def make_minigrid_env(name='MiniGrid-Empty-5x5-v0'):
    env = gym.make(name)

    env = gym_minigrid.wrappers.FlatObsWrapper(env)

    return env


class Tabular_Minigrid():
    def __init__(self, env):

        self.env = env
        self.state_shape = env.observation_space.shape or env.observation_space.n

        self.height = env.env.grid.height - 2  # assume square grid always
        self.actions = [0, 1, 2]  # hardcoded, left/right/forwad

        self.__init_q_states__()

    def __init_q_states__(self):

        # grid*grid*possible directions * actions
        # smallest -> 5*5*4*3 = 300
        self.q = torch.zeros(self.height, self.height, 4, len(self.actions))
        self.state_obs = torch.zeros(self.height, self.height, 4,
                                     np.prod(self.state_shape))  # testing in size, otherwise

        self.nr_states = self.height * self.height * 4

    def travel_state_actions(self, policy):



        # go over all possible states and actions
        for h in range(self.height):
            for w in range(self.height):
                for dir in range(4):  # every direction

                    # # run less
                    # if random.random() > (200/self.nr_states):
                    #     break

                    for a in self.actions:
                        # for j in range(1): #rollouts per state-pair wise, deterministic so 1

                        self.env.reset()
                        self.set_minigrid_env_state((h, w), dir)

                        if a == 0:  # only do it once
                            # get state observation, by perfoming done action
                            obs, _, _, _ = self.env.step(6)
                            # save initial observation state for later
                            self.state_obs[h, w, dir] = torch.from_numpy(obs.reshape(-1))

                        # perform exploration action step in dpi algorightm
                        obs, r, d, _ = self.env.step(a)

                        # FIXME obs probably needs batch, almost for sure

                        self.q[h, w, dir, a] += self.perform_rollout(obs, r, policy)

    # def get_possible_states_list(self):
    #     # dont love this, but for shuffling
    #     states = np.zeros((self.height, self.height, 4, 3))
    #
    #     for h in range(self.height):
    #         for w in range(self.height):
    #             for dir in range(4):  # every direction
    #                 states[h, w, dir] = [h, w, dir]
    #
    #     return states.reshape(-1, 3)

    def get_train_batch(self, batch=64):

        assert batch > 0, "batch =< 0"

        # states = self.get_possible_states_list()
        # np.random.shuffle(states) # performance.... shufles in dim 0
        # nr_states = states.shape[0]
        #
        # for i in range(0,nr_states,batch):
        #

        # diff aproach

        q = self.q.view(-1, len(self.actions))
        state_obs = self.state_obs.view(-1, np.prod(self.state_shape))

        assert q.shape[0] == state_obs.shape[0], "q vs state_obs shape"

        indexes = np.arange(0, self.nr_states)
        np.random.shuffle(indexes)

        for i in range(0, self.nr_states, batch):
            yield q[indexes[i:i + batch]], state_obs[indexes[i:i + batch]]

    def perform_rollout(self, obs, r, policy, horizon=100):

        gamma = policy._gamma  # starts to show bad code/abstraction

        q_value = r

        for i in range(1, horizon + 1):

            action = policy.forward([obs])
            obs, r, done, _ = self.env.step(action)

            q_value += r * (gamma ** i)
            if done:
                break

        return q_value

    def set_minigrid_env_state(self, agent_pos: tuple, dir: int):
        assert dir >= 0 and dir < 4, "Minigrid state set, invalid agent direction"

        # correct tuples
        agent_pos = (agent_pos[0] + 1, agent_pos[1] + 1)

        assert isinstance(agent_pos, tuple) and len(agent_pos) == 2 and agent_pos[0] < self.height + 1 \
               and agent_pos[1] < self.height + 1 and agent_pos[0] > 0 and agent_pos[
                   1] > 0, "Minigrid state set, invalid agent direction"

        self.env.env.agent_pos = agent_pos
        self.env.env.agent_dir = dir
