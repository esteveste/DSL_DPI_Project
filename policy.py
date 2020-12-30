import torch
from torch import nn
import numpy as np
from copy import deepcopy
from typing import Any, Dict, Union, Optional

# from tianshou.policy import BasePolicy
from tianshou.data import Batch, ReplayBuffer, to_torch_as, to_numpy

import utils


class DPI():
    def __init__(self,
                 model: torch.nn.Module,
                 optim: torch.optim.Optimizer,
                 discount_factor: float = 0.99,
                 # reward_normalization: bool = False,
                 train_epochs=10,
                 batch_size=16,
                 **kwargs: Any, ):

        self.model = model
        self.optim = optim
        self.eps = 0.0
        assert (
                0.0 <= discount_factor <= 1.0
        ), "discount factor should be in [0, 1]"
        self._gamma = discount_factor
        # assert estimation_step > 0, "estimation_step should be greater than 0"

        # self._rew_norm = reward_normalization

        self.epochs = train_epochs
        self.batch_size = batch_size

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, obs):

        q, _ = self.model(obs)

        # something like this, on random exploration?
        # q=q+torch.rand_like(q)*0.1

        # epsilon greedy
        # boltzman

        # torch.softmax(q) #[0.249999999,0.250000001,0.24999999,0.249999999] #[0,0,0,0]
        # q =to_numpy(torch.softmax(q[0],dim=-1))
        # action = np.random.choice(np.arange(q.shape[-1]),p=q)

        action = to_numpy(q.max(dim=1)[1])  # choose max q

        return action

    def learn(self, tabular_env: utils.Tabular_Minigrid):

        # batch learning
        print("Policy Training")

        nr_states = tabular_env.nr_states

        for e in range(self.epochs):
            total_loss = 0
            for qs, obs in tabular_env.get_train_batch(self.batch_size):
                self.optim.zero_grad()

                # just copy q for now
                pred_q, _ = self.model(obs)

                # target = torch.argmax(qs, dim=1)  # target is max q

                # if same q value, chooses target at random
                target = torch.argmax(torch.rand(*qs.shape) * (qs == qs.max(dim=1, keepdims=True)[0]), dim=1)
                loss = self.criterion(pred_q, target)

                loss.backward()

                self.optim.step()

                # statistics
                # pred_target = pred_q.argmax(dim=1)

                total_loss += loss.item()

            print(f"Epoch {e:2d}, Loss {total_loss / nr_states:.4f}")

        return loss

##### old tianshou code

# class DPI(BasePolicy):
#     """Based on Deep Q Class
#
#         DPI needs tabular Q states, enviroment state reset
#
#         Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
#         explanation.
#     """
#
#     def __init__(
#         self,
#         model: torch.nn.Module,
#         optim: torch.optim.Optimizer,
#         discount_factor: float = 0.99,
#         # estimation_step: int = 1,
#         # target_update_freq: int = 0,
#         reward_normalization: bool = False,
#         **kwargs: Any,
#     ) -> None:
#
#         super().__init__(**kwargs)
#         self.model = model
#         self.optim = optim
#         self.eps = 0.0
#         assert (
#             0.0 <= discount_factor <= 1.0
#         ), "discount factor should be in [0, 1]"
#         self._gamma = discount_factor
#         # assert estimation_step > 0, "estimation_step should be greater than 0"
#
#         # self._n_step = estimation_step
#         # self._target = target_update_freq > 0
#         # self._freq = target_update_freq
#         # self._cnt = 0
#         # if self._target:
#         #     self.model_old = deepcopy(self.model)
#         #     self.model_old.eval()
#         self._rew_norm = reward_normalization
#
#     # def set_eps(self, eps: float) -> None:
#     #     """Set the eps for epsilon-greedy exploration."""
#     #     self.eps = eps
#
#     def train(self, mode: bool = True) -> "DQNPolicy":
#         """Set the module in training mode, except for the target network."""
#         self.training = mode
#         self.model.train(mode)
#         return self
#
#     # def sync_weight(self) -> None:
#     #     """Synchronize the weight for the target network."""
#     #     self.model_old.load_state_dict(self.model.state_dict())
#
#     # def _target_q(
#     #     self, buffer: ReplayBuffer, indice: np.ndarray
#     # ) -> torch.Tensor:
#     #     batch = buffer[indice]  # batch.obs_next: s_{t+n}
#     #     if self._target:
#     #         # target_Q = Q_old(s_, argmax(Q_new(s_, *)))
#     #         a = self(batch, input="obs_next").act
#     #         with torch.no_grad():
#     #             target_q = self(
#     #                 batch, model="model_old", input="obs_next"
#     #             ).logits
#     #         target_q = target_q[np.arange(len(a)), a]
#     #     else:
#     #         with torch.no_grad():
#     #             target_q = self(batch, input="obs_next").logits.max(dim=1)[0]
#     #     return target_q
#
#     def process_fn(
#         self, batch: Batch, buffer: ReplayBuffer, indice: np.ndarray
#     ) -> Batch:
#         """Compute the n-step return for Q-learning targets.
#
#         More details can be found at
#         :meth:`~tianshou.policy.BasePolicy.compute_nstep_return`.
#         """
#         batch = self.compute_nstep_return(
#             batch, buffer, indice, self._target_q,
#             self._gamma, self._n_step, self._rew_norm)
#         return batch
#
#     def forward(
#         self,
#         batch: Batch,
#         state: Optional[Union[dict, Batch, np.ndarray]] = None,
#         model: str = "model",
#         input: str = "obs",
#         **kwargs: Any,
#     ) -> Batch:
#         """Compute action over the given batch data.
#
#         If you need to mask the action, please add a "mask" into batch.obs, for
#         example, if we have an environment that has "0/1/2" three actions:
#         ::
#
#             batch == Batch(
#                 obs=Batch(
#                     obs="original obs, with batch_size=1 for demonstration",
#                     mask=np.array([[False, True, False]]),
#                     # action 1 is available
#                     # action 0 and 2 are unavailable
#                 ),
#                 ...
#             )
#
#         :param float eps: in [0, 1], for epsilon-greedy exploration method.
#
#         :return: A :class:`~tianshou.data.Batch` which has 3 keys:
#
#             * ``act`` the action.
#             * ``logits`` the network's raw output.
#             * ``state`` the hidden state.
#
#         .. seealso::
#
#             Please refer to :meth:`~tianshou.policy.BasePolicy.forward` for
#             more detailed explanation.
#         """
#         model = getattr(self, model)
#         obs = batch[input]
#         obs_ = obs.obs if hasattr(obs, "obs") else obs
#         q, h = model(obs_, state=state, info=batch.info)
#         act: np.ndarray = to_numpy(q.max(dim=1)[1])  # choose max q
#
#         # masking could be cool
#         # if hasattr(obs, "mask"):
#         #     # some of actions are masked, they cannot be selected
#         #     q_: np.ndarray = to_numpy(q)
#         #     q_[~obs.mask] = -np.inf
#         #     act = q_.argmax(axis=1)
#
#
#         # EXPLORATION ACTION NOISE
#         # # add eps to act in training or testing phase
#         # if not self.updating and not np.isclose(self.eps, 0.0):
#         #     for i in range(len(q)):
#         #         if np.random.rand() < self.eps:
#         #             q_ = np.random.rand(*q[i].shape)
#         #             if hasattr(obs, "mask"):
#         #                 q_[~obs.mask[i]] = -np.inf
#         #             act[i] = q_.argmax()
#
#         return Batch(logits=q, act=act, state=h)
#
#     def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
#         # if self._target and self._cnt % self._freq == 0:
#         #     self.sync_weight()
#         #
#         self.optim.zero_grad()
#         weight = batch.pop("weight", 1.0)
#         q = self(batch).logits
#         q = q[np.arange(len(q)), batch.act]
#         r = to_torch_as(batch.returns.flatten(), q)
#         td = r - q
#         loss = (td.pow(2) * weight).mean()
#         batch.weight = td  # prio-buffer
#         loss.backward()
#         self.optim.step()
#         self._cnt += 1
#
#         return {"loss": loss.item()}
