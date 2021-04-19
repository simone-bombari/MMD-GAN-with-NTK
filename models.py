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


class Convolutional(nn.Module):
    def __init__(self):
        super(Convolutional, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)  # 1 channel for MNIST
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension (MNIST)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)  # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))  # Not putting MNIST dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class FullyConnectedClassifier(nn.Module):
    def __init__(self):
        super(FullyConnectedClassifier, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
