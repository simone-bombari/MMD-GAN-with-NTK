import torch
from torch import nn
from torch.nn import functional as F


class ConvTranspose(nn.Module):  # It doesn't work...
    def __init__(self, latent_size):
        super(ConvTranspose, self).__init__()
        self.fc1 = nn.Linear(latent_size, 64)
        self.fc2 = nn.Linear(64, 12 * 12)
        self.conv_tr_1 = nn.ConvTranspose2d(1, 5, 4, stride=2)  # in_channels, out_channels, kernel_size, stride
        self.conv_tr_2 = nn.ConvTranspose2d(5, 10, 3, stride=2)
        self.conv_1 = nn.Conv2d(10, 1, 3, padding=1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = x.reshape((-1, 1, 16, 16))
        x = F.relu(self.conv_tr_1(x))
        x = F.relu(self.conv_tr_2(x))
        x = torch.sigmoid(self.conv_1(x))
        return x


class FullyConnected(nn.Module):  # An easy fully connected that does its job
    def __init__(self, latent_size):
        super(FullyConnected, self).__init__()
        self.fc1 = nn.Linear(latent_size, 64)
        self.fc2 = nn.Linear(64, 256)
        self.fc3 = nn.Linear(256, 28 * 28)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        x = x.reshape((-1, 1, 28, 28))
        return x
