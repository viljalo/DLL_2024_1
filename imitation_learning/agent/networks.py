import torch.nn as nn
import torch
import torch.nn.functional as F

"""
Imitation learning network
"""


class CNN(nn.Module):

    def __init__(self, history_length=3, n_classes=5):
        super(CNN, self).__init__()
        # TODO : define layers of a convolutional neural network
        self.conv1 = nn.Conv2d(history_length, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(8 * 8 * 64, 256)
        self.fc3 = nn.Linear(256, n_classes)  # 5 possible action id's (discretized)

    def forward(self, x):
        # TODO: compute forward pass
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc3(x)
        return x
