import torch
import torch.nn as nn

class GomokuNet(nn.Module):
    def __init__(self, board_size=19):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * board_size * board_size, 512)
        self.fc2 = nn.Linear(512, board_size * board_size)
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
