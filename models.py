from torch import nn
from torch.nn import functional as F


class Reshape(nn.Module):
    '''
        Used in a nn.Sequential pipeline to reshape on the fly.
    '''

    def __init__(self, *target_shape):
        super().__init__()
        self.target_shape = target_shape

    def forward(self, x):
        return x.view(*self.target_shape)


class MMD_VAE(nn.Module):
    def __init__(self, latent_size):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=5, kernel_size=5, padding=2),  # 1*28*28 -> 5*28*28
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=5, out_channels=5, kernel_size=5),  # 5*28*28 -> 5*24*24
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=5, out_channels=5, kernel_size=5),  # 5*24*24 -> 5*20*20
            nn.LeakyReLU(),
            Reshape([-1, 5 * 20 * 20]),
            nn.Linear(in_features=5 * 20 * 20, out_features=5 * 12),
            nn.LeakyReLU(),
            nn.Linear(in_features=5 * 12, out_features=latent_size)
        )

        self.decoder = nn.Sequential(
            nn.Linear(in_features=latent_size, out_features=5 * 12),
            nn.ReLU(),
            nn.Linear(in_features=5 * 12, out_features=24 * 24),
            nn.ReLU(),
            Reshape([-1, 1, 24, 24]),
            nn.ConvTranspose2d(in_channels=1, out_channels=5, kernel_size=3),  # 1*24*24 -> 5*26*26
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=5, out_channels=10, kernel_size=5),  # 5*26*26 -> 10*30*30
            nn.ReLU(),
            nn.Conv2d(in_channels=10, out_channels=1, kernel_size=3),  # 10*30*30 -> 1*28*28
            nn.Sigmoid()
        )

    def forward(self, X):
        if self.training:
            latent = self.encoder(X)
            return self.decoder(latent), latent
        else:
            return self.decoder(self.encoder(X))


class mmd_generator(nn.Module):
    def __init__(self, latent_size):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(in_features=latent_size, out_features=5 * 12),
            nn.ReLU(),
            nn.Linear(in_features=5 * 12, out_features=24 * 24),
            nn.ReLU(),
            Reshape([-1, 1, 24, 24]),
            nn.ConvTranspose2d(in_channels=1, out_channels=5, kernel_size=3),  # 1*24*24 -> 5*26*26
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=5, out_channels=10, kernel_size=5),  # 5*26*26 -> 10*30*30
            nn.ReLU(),
            nn.Conv2d(in_channels=10, out_channels=1, kernel_size=3),  # 10*30*30 -> 1*28*28
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.decoder(z)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 3)  # 1 channel for MNIST
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension (MNIST)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
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
