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


class ConvNet(nn.Module):
    # ARCH BASED IN https://github.com/lcswillems/rl-starter-files/blob/master/model.py
    def __init__(self, state_shape, action_shape):
        super().__init__()

        assert len(state_shape)==3, "state space not a 2D image"

        n = state_shape[0]
        m = state_shape[1]
        self.image_embedding_size = ((n-1)//2-2)*((m-1)//2-2)*64

        self.model = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
            nn.Linear(self.image_embedding_size, 128),
            nn.ReLU(),
            nn.Linear(128, np.prod(action_shape)),
        )



    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float)

        assert len(obs.shape) >= 2, "obs shape is less than 2 dim, has batch?"

        batch = obs.shape[0]
        logits = self.model(obs)
        return logits, state
