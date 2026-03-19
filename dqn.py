import torch
from torch import nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, enable_dueling=True):
        super(DQN, self).__init__()
        self.enable_dueling = enable_dueling
        
        # Shared feature extractor
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )

        if self.enable_dueling:
            # Value stream: V(s)
            self.value_stream = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
            # Advantage stream: A(s, a)
            self.advantage_stream = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim)
            )
        else:
            self.output = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        features = self.feature_layer(x)
        if self.enable_dueling:
            v = self.value_stream(features)
            a = self.advantage_stream(features)
            return v + (a - a.mean(dim=1, keepdim=True))
        return self.output(features)