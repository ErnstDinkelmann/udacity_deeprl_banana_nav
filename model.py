import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, int_hl_1_num_units=32, int_hl_2_num_units=32):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        # Defining the layers
        self.fc1 = nn.Linear(state_size, int_hl_1_num_units, bias=True)
        self.fc2 = nn.Linear(int_hl_1_num_units, int_hl_2_num_units, bias=True)
        self.fc3 = nn.Linear(int_hl_2_num_units, action_size)

    # def forward(self, state):
    def forward(self, x):
        """Build a network that maps state -> action values."""
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x
