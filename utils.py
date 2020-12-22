import gym
import gym_minigrid

import torch
import tianshou
import numpy as np

# gym_minigrid.wrappers.FlatObsWrapper

def make_minigrid_env(name):

    env = gym.make(name)

    env = gym_minigrid.wrappers.FlatObsWrapper(env)

    return env



class Tabular_Minigrid():
    def __init__(self,env):

        self.env = env

        self.height = env.env.grid.height # assume square grid always

        self.actions = [0,1,2] # hardcoded, left/right/forwad

        self.__init_q_states__()

    def __init_q_states__(self):

        # grid*grid*possible directions * actions
        # smallest -> 5*5*4*3 = 300
        self.q = torch.zeros(self.height,self.height,4,len(self.actions))

    def travel_state_actions(self,policy):

        # go over all possible states and actions
        for h in self.height:
            for w in self.height:
                for dir in range(4): #every direction
                    for a in self.actions:

                        self.env.reset()

                        self.set_minigrid_env_state((h,w),)

    # def rollout_from_state



    def set_minigrid_env_state(self,agent_pos:tuple,dir:int):
        assert dir >= 0 and dir < 4, "Minigrid state set, invalid agent direction"
        assert isinstance(agent_pos,tuple) and len(agent_pos)==2 and agent_pos[0]<self.grid_size\
               and agent_pos[1]<self.grid_size, "Minigrid state set, invalid agent direction"

        env.env.agent_pos = agent_pos
        env.env.agent_dir = dir