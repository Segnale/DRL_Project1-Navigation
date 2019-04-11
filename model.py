import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=32):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(DQN, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class Duel_DQN(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=32):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Duel_DQN, self).__init__()
        self.action_size = action_size
        self.seed = torch.manual_seed(seed)

        self.fc1_stt = nn.Linear(state_size, fc1_units)
        self.fc1_act = nn.Linear(state_size, fc1_units)

        self.fc2_stt = nn.Linear(fc1_units, fc2_units)
        self.fc2_act = nn.Linear(fc1_units, fc2_units)

        self.fc3_stt = nn.Linear(fc2_units, 1)
        self.fc3_act = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        stt = F.relu(self.fc1_stt(state))
        act = F.relu(self.fc1_act(state))

        stt = F.relu(self.fc2_stt(stt))
        act = F.relu(self.fc2_act(act))

        stt = F.relu(self.fc3_stt(stt))
        act = F.relu(self.fc3_act(act))

        return stt + act - act.mean(1).unsqueeze(1).expand(state.size(0), self.action_size)
