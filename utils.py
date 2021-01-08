import gym
import gym_minigrid

import torch
import tianshou
import numpy as np

import random

from custom_env import *

import pdb

# gym_minigrid.wrappers.FlatObsWrapper

def make_minigrid_env(name='MiniGrid-Empty-5x5-v0', flatten=True):
    env = gym.make(name)

    if flatten:
        env = gym_minigrid.wrappers.FlatObsWrapper(env)
    else:
        env = gym_minigrid.wrappers.ImgObsWrapper(env)

    return env

def make_CartPole_env():
    env = CartPoleEnv()
    return env

class Tabular_Minigrid():
    def __init__(self, env,sample_n_states=1000):

        self.sample_n_states = sample_n_states

        self.env = env
        self.state_shape = env.observation_space.shape or env.observation_space.n

        self.height = env.env.grid.height - 2  # assume square grid always

        # configure forced actions in rollout
        env_name = env.spec.id
        if "Empty" in env_name:
            self.actions = [0, 1, 2]  # hardcoded, left/right/forward
        else:
            self.actions = [0, 1, 2, 3, 4, 5]

        self.__init_q_states__()

    def __init_q_states__(self):

        # grid*grid*possible directions * actions
        # smallest -> 5*5*4*3 = 300
        self.q = torch.zeros(self.height, self.height, 4, len(self.actions))
        self.q += -1
        self.state_obs = torch.zeros(self.height, self.height, 4,
                                     np.prod(self.state_shape))  # testing in size, otherwise

        self.nr_states = self.height * self.height * 4

    # sets the env to the correct gym, used in travel_state_actions
    def set_minigrid_env_state(self, agent_pos: tuple, dir: int):
        assert dir >= 0 and dir < 4, "Minigrid state set, invalid agent direction"

        # correct tuples
        agent_pos = (agent_pos[0] + 1, agent_pos[1] + 1)

        assert isinstance(agent_pos, tuple) and len(agent_pos) == 2 and agent_pos[0] < self.height + 1 \
               and agent_pos[1] < self.height + 1 and agent_pos[0] > 0 and agent_pos[
                   1] > 0, "Minigrid state set, invalid agent direction"

        self.env.env.agent_pos = agent_pos
        
        
        self.env.env.agent_dir = dir

    # performs the rollout and calculates the qs with the policy
    # after selected on travel_state_actions()
    def perform_rollout(self, obs, r, policy, horizon=50):

        gamma = policy._gamma  # starts to show bad code/abstraction

        q_value = r

        observations = []
        actions = []
        rewards = []

        for i in range(1, horizon + 1):

            action = policy.forward([obs])
            obs, r, done, _ = self.env.step(action)
            
            observations += [obs]
            actions += [action]
            rewards += [r]

            q_value += r * (gamma ** i)
            if done:
                break

        return q_value, observations, actions, rewards

    def random_state(self,N):

        for n in range(N):
            yield np.random.choice(range(self.height)),np.random.choice(range(self.height)),np.random.choice(range(4))


    # goes over every state-action pair of the environment
    def travel_state_actions(self, policy):

        # go over all possible states and actions
        # for h in range(self.height):
        #     for w in range(self.height):
        #         for dir in range(4):  # every direction
        for h,w,dir in self.random_state(N=self.sample_n_states):

            # # # run less
            # if random.random() > (200/self.nr_states):
            #     break

            for a in self.actions:
                # for j in range(1): #rollouts per state-pair wise, deterministic so 1

                self.env.reset()
                self.set_minigrid_env_state((h, w), dir)

                # self.env.render()

                if a == 0:  # only do it once
                    # get state observation, by perfoming useless action (done action)
                    obs, _, _, _ = self.env.step(6)
                    # save initial observation state for later
                    self.state_obs[h, w, dir] = torch.from_numpy(obs.reshape(-1))

                # perform exploration action step in dpi algorightm
                obs, r, d, _ = self.env.step(a)

                self.q[h, w, dir, a], observations, actions, rewards = self.perform_rollout(obs, r, policy) # FIXME FAZER A MEDIA
                horizon_range = len(observations)
                for k, observation in enumerate(observations):
                    h = int(observation[0])
                    w = int(observation[1])
                    direct = int(observation[2])
                    action = actions[k]
                    reward = 0
                    if self.q[h, w, direct, action] == -1:
                        for l in range(k, horizon_range):
                            reward += rewards[l] * gamma**(l-k)
                        self.q[h, w, direct, action] = reward
                        self.state_obs[h, w, direct] = torch.from_numpy(observation.reshape(-1))

    # gets a shufled training batch from the q table, to be used
    # on the policy.learn()
    def get_train_batch(self, batch=8):

        assert batch > 0, "batch =< 0"

        q = self.q.view(-1, len(self.actions)) # shape -> nr_states,actions
        state_obs = self.state_obs.view(-1, np.prod(self.state_shape))

        assert q.shape[0] == state_obs.shape[0], "q vs state_obs shape"

        # indexes = np.arange(0, self.nr_states)

        ## valid indexes, ve se estados nao tem 1a acao com q com -1
        arr = np.sign(q[:, 0] + 0.01)
        for a in range(1, len(self.actions)):
            arr = arr + np.sign(q[:, a] + 0.01)
        
        indexes = np.where(arr[:] != -1)[0]

        np.random.shuffle(indexes)

        for i in range(0, len(indexes), batch):
            yield q[indexes[i:i + batch]], state_obs[indexes[i:i + batch]]

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
class Tabular_CartPole():
    def __init__(self, env, sample_n_states=1000, disc = 50):
                
        self.sample_n_states = sample_n_states

        self.env = env
        self.state_shape = env.observation_space.shape or env.observation_space.n

        #self.height = env.env.grid.height - 2  # assume square grid always
        self.disc = disc

        
        # configure forced actions in rollout
        # env_name = env.spec.id
        # if "Empty" in env_name:
        #     self.actions = [0, 1, 2]  # hardcoded, left/right/forward
        # else:
        #     self.actions = [0, 1, 2, 3, 4, 5]
        self.actions = [0, 1]

        self.__init_q_states__()

    def __init_q_states__(self):

        # grid*grid*possible directions * actions
        # smallest -> 5*5*4*3 = 300
        self.q = torch.zeros(self.disc, self.disc, self.disc, self.disc, len(self.actions))
        self.q += -1
        self.state_obs = torch.zeros(self.disc, self.disc, self.disc, self.disc, np.prod(self.state_shape))  # testing in size, otherwise

        self.nr_states = self.disc * self.disc * self.disc * self.disc

    def to_ind(self, pos, vel, pos_ang, vel_ang):
        
        pos_cand = int(np.floor(self.disc * (pos + 2.4) / 4.8))
        if pos_cand < 0:
            pos_ind = 0
        if pos_cand >= self.disc:
            pos_ind = self.disc - 1
        if pos_cand >= 0 and pos_cand < self.disc:
            pos_ind = pos_cand
            
        vel_cand = int(np.floor(self.disc * (vel + 5) / 10))
        if vel_cand < 0:
            vel_ind = 0
        if vel_cand >= self.disc:
            vel_ind = self.disc - 1
        if vel_cand >= 0 and vel_cand < self.disc:
            vel_ind = vel_cand
        
        pos_ang_cand = int(np.floor(self.disc * (pos_ang + 0.209) / 0.418))
        if pos_ang_cand < 0:
            pos_ang_ind = 0
        if pos_ang_cand >= self.disc:
            pos_ang_ind = self.disc - 1
        if pos_ang_cand >= 0 and pos_ang_cand < self.disc:
            pos_ang_ind = pos_ang_cand
        
        vel_ang_cand = int(np.floor(self.disc * (vel_ang + 5) / 10))
        if vel_ang_cand < 0:
            vel_ang_ind = 0
        if vel_ang_cand >= self.disc:
            vel_ang_ind = self.disc - 1
        if vel_ang_cand >= 0 and vel_ang_cand < self.disc:
            vel_ang_ind = vel_ang_cand
        
        return pos_ind, vel_ind, pos_ang_ind, vel_ang_ind
        
        
    # # sets the env to the correct gym, used in travel_state_actions
    # def set_minigrid_env_state(self, agent_pos: tuple, dir: int):
    #     assert dir >= 0 and dir < 4, "Minigrid state set, invalid agent direction"

    #     # correct tuples
    #     agent_pos = (agent_pos[0] + 1, agent_pos[1] + 1)

    #     assert isinstance(agent_pos, tuple) and len(agent_pos) == 2 and agent_pos[0] < self.height + 1 \
    #             and agent_pos[1] < self.height + 1 and agent_pos[0] > 0 and agent_pos[
    #                 1] > 0, "Minigrid state set, invalid agent direction"

    #     self.env.env.agent_pos = agent_pos
    #     self.env.env.agent_dir = dir

    # performs the rollout and calculates the qs with the policy
    # after selected on travel_state_actions()
    def perform_rollout(self, obs, r, policy, horizon=50):

        gamma = policy._gamma  # starts to show bad code/abstraction

        q_value = r
        
        observations = []
        actions = []
        rewards = []
        
        for i in range(1, horizon + 1):

            action = policy.forward([obs])[0]
            obs, r, done, _ = self.env.step(action)
            
            observations += [obs]
            actions += [action]
            rewards += [r]
            
            q_value += r * (gamma ** i)
            if done:
                break

        return q_value, observations, actions, rewards

    def random_state(self,N):

        for n in range(N):
            rand = np.random.rand(4)
            
            pos = rand[0] * 4.8 - 2.4
            vel = rand[1] * 10 - 5
            pos_ang = rand[2] * 0.418 - .209
            vel_ang = rand[3]* 10 - 5
            
            yield pos, vel, pos_ang, vel_ang


    # goes over every state-action pair of the environment
    def travel_state_actions(self, policy):
        # go over all possible states and actions
        # for h in range(self.height):
        #     for w in range(self.height):
        #         for dir in range(4):  # every direction
        gamma = policy._gamma
        for pos, vel, pos_ang, vel_ang in self.random_state(N=self.sample_n_states):

            # # # run less
            # if random.random() > (200/self.nr_states):
            #     break

            for a in self.actions:
                # for j in range(1): #rollouts per state-pair wise, deterministic so 1

                self.env.reset(state = (pos, vel, pos_ang, vel_ang))
                #self.set_minigrid_env_state((h, w), dir)

                # self.env.render()

                # if a == 0:  # only do it once
                #     # get state observation, by perfoming useless action (done action)
                #     obs, _, _, _ = self.env.step(6)
                #     # save initial observation state for later
                #     self.state_obs[h, w, dir] = torch.from_numpy(obs.reshape(-1))

                # perform exploration action step in dpi algorightm

                obs, r, d, _ = self.env.step(a)
                pos_ind, vel_ind, pos_ang_ind, vel_ang_ind = self.to_ind(pos, vel, pos_ang, vel_ang)
                if d != True:
                    self.q[pos_ind, vel_ind, pos_ang_ind, vel_ang_ind, a], observations, actions, rewards = self.perform_rollout(obs, r, policy) # FIXME FAZER A MEDIA
                else:
                    self.q[pos_ind, vel_ind, pos_ang_ind, vel_ang_ind, a] = 1
                    observations = []
                    actions = []
                    rewards = []
                self.state_obs[pos_ind, vel_ind, pos_ang_ind, vel_ang_ind] = torch.tensor([pos, vel, pos_ang, vel_ang])
                
                
                horizon_range = len(observations)
                for k, observation in enumerate(observations):
                    pos = observation[0]
                    vel = observation[1]
                    pos_ang = observation[2]
                    vel_ang = observation[3]
                    pos_ind, vel_ind, pos_ang_ind, vel_ang_ind = self.to_ind(pos, vel, pos_ang, vel_ang)
                    action = actions[k]
                    reward = 0
                    if self.q[pos_ind, vel_ind, pos_ang_ind, vel_ang_ind, action] == -1:
                        for l in range(k, horizon_range):
                            reward += rewards[l] * gamma**(l-k)
                        self.q[pos_ind, vel_ind, pos_ang_ind, vel_ang_ind, action] = reward
                        self.state_obs[pos_ind, vel_ind, pos_ang_ind, vel_ang_ind] = torch.tensor([pos, vel, pos_ang, vel_ang])
    # gets a shufled training batch from the q table, to be used
    # on the policy.learn()
    def get_train_batch(self, batch=8):
        
        #pdb.set_trace()
        
        assert batch > 0, "batch =< 0"

        q = self.q.view(-1, len(self.actions)) # shape -> nr_states,actions
        state_obs = self.state_obs.view(-1, np.prod(self.state_shape))

        assert q.shape[0] == state_obs.shape[0], "q vs state_obs shape"

        # indexes = np.arange(0, self.nr_states)

        ## valid indexes, ve se estados nao tem 1a acao com q com -1
        indexes = np.where(np.sign(q[:,0] + 0.01) + np.sign(q[:, 1] + 0.01) > 0)[0]
        np.random.shuffle(indexes)
        for i in range(0, len(indexes), batch):
            yield q[indexes[i:i + batch]], state_obs[indexes[i:i + batch]]
