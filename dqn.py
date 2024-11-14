import torch
from torch import nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=256):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, output_size)
        self.fc2 = nn.Linear(output_size, hidden_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

if __name__ == "__main__":
    input_size = 12  # Assuming the input is a 8-dimensional vector
    hidden_size = 2
    output_size = 256
    net = DQN(input_size, hidden_size, output_size)
    state = torch.randn(1, input_size)
    output = net(state)
    print(output)