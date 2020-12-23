import os
import pprint
import argparse

import gym
import gym_minigrid
import torch
import tianshou
import numpy as np

from torch.utils.tensorboard import SummaryWriter

from tianshou.policy import DQNPolicy
from tianshou.env import SubprocVectorEnv
from tianshou.utils.net.discrete import DQN
from tianshou.trainer import offpolicy_trainer
from tianshou.data import Collector, ReplayBuffer

from policy import DPI
from network import MLP
from utils import *


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='MiniGrid-Empty-Random-6x6-v0')
    # parser.add_argument('--task', type=str, default='MiniGrid-Empty-16x16-v0')
    parser.add_argument('--seed', type=int, default=0)
    # parser.add_argument('--eps-test', type=float, default=0.005)
    # parser.add_argument('--eps-train', type=float, default=1.)
    # parser.add_argument('--eps-train-final', type=float, default=0.05)
    parser.add_argument('--buffer-size', type=int, default=100000)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--gamma', type=float, default=0.99)
    # parser.add_argument('--n-step', type=int, default=3)
    # parser.add_argument('--target-update-freq', type=int, default=500)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--policy-epoch', type=int, default=4)
    # parser.add_argument('--step-per-epoch', type=int, default=10000)
    parser.add_argument('--collect-per-step', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--training-num', type=int, default=16)
    parser.add_argument('--test-num', type=int, default=10)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument(
        '--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--frames_stack', type=int, default=4)
    parser.add_argument('--resume_path', type=str, default=None)
    parser.add_argument('--watch', default=False, action='store_true',
                        help='watch the play of pre-trained policy only')
    return parser.parse_args()


def train_dpi(args):
    env = make_minigrid_env(args.task)

    state_shape = env.observation_space.shape or env.observation_space.n
    # action_shape = env.env.action_space.shape or env.env.action_space.n
    action_shape = 3  # selecting Basic actions in minigrid

    print("Observations shape:", state_shape)
    print("Actions shape:", action_shape)

    # seed FIXME
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env.seed(args.seed)
    # train_envs.seed(args.seed)
    # test_envs.seed(args.seed)

    net = MLP(state_shape, action_shape)

    optim = torch.optim.Adam(net.parameters(), lr=args.lr)

    # define policy
    policy = DPI(net, optim, discount_factor=0.99)

    # # load a previous policy
    if args.resume_path:
        policy.load_state_dict(torch.load(
            args.resume_path, map_location=args.device
        ))
        print("Loaded agent from: ", args.resume_path)

    if "MiniGrid" in args.task:
        tabular_env = Tabular_Minigrid(env)

    print(f"Num states: {tabular_env.nr_states}")

    # Training loop
    for e in range(args.epoch):
        tabular_env.__init_q_states__()

        # collect qs
        tabular_env.travel_state_actions(policy)

        # learn new policy
        policy.learn(tabular_env)

        scores = evaluate(policy, env, runs=10)

        print(f"Eval Score: {scores.mean():.2f} +- {scores.std():.2f} Total registered q:", tabular_env.q.sum().item())


def evaluate(policy, env, runs=2):
    # simple evaluation, from init state
    total_scores = np.zeros(runs)

    for i in range(runs):
        done = False
        obs = env.reset()
        ep_reward=0

        while not done:
            action = policy.forward([obs])
            obs, r, done, _ = env.step(action)
            ep_reward += r

        total_scores[i] = ep_reward

    return total_scores


if __name__ == "__main__":
    train_dpi(get_args())
