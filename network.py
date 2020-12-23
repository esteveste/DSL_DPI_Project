import torch, numpy as np
from torch import nn


class MLP(nn.Module):
    def __init__(self, state_shape, action_shape):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(np.prod(state_shape), 128), nn.ReLU(inplace=True),
            # nn.Linear(128, 128), nn.ReLU(inplace=True),
            # nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Linear(128, np.prod(action_shape)),
        )

    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float)

        assert len(obs.shape) >= 2, "obs shape is less than 2 dim, has batch?"

        batch = obs.shape[0]
        logits = self.model(obs.view(batch, -1))
        return logits, state
