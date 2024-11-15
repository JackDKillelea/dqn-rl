import torch
from torch import nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=256, enable_dueling_dqn=True):
        super(DQN, self).__init__()
        self.enable_dueling_dqn = enable_dueling_dqn
        self.fc1 = nn.Linear(input_size, output_size)

        if self.enable_dueling_dqn:
            # value stream
            self.fc_value = nn.Linear(output_size, 256)
            self.value = nn.Linear(256, 1)

            # Advantage stream
            self.fc_advantage = nn.Linear(output_size, 256)
            self.advantages = nn.Linear(256, hidden_size)
        else:
            self.fc2 = nn.Linear(output_size, hidden_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))

        if self.enable_dueling_dqn:
            # Value calculation
            value = F.relu(self.fc_value(x))
            v = self.value(value)

            # Advantage calculation
            advantage = F.relu(self.fc_advantage(x))
            a = self.advantages(advantage)

            # Calculate Q-values by adding value and advantage streams and subtracting mean advantage
            q_values = v + a - torch.mean(a, dim=1, keepdim=True)
            return q_values
        else:
            return self.fc2(x)

if __name__ == "__main__":
    input_size = 12  # Assuming the input is a 8-dimensional vector
    hidden_size = 2
    output_size = 256
    net = DQN(input_size, hidden_size, output_size)
    state = torch.randn(1, input_size)
    output = net(state)
    print(output)