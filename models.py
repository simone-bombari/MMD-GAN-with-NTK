from torch import nn
from torch.nn import functional as F


class Reshape(nn.Module):
    '''
        code from https://github.com/Saswatm123/MMD-VAE/blob/master/MMD_VAE.ipynb
    '''

    def __init__(self, *target_shape):
        super().__init__()
        self.target_shape = target_shape

    def forward(self, x):
        return x.view(*self.target_shape)


class ConvTranspose(nn.Module):
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


class FullyConnected(nn.Module):
    def __init__(self, latent_size):
        super(FullyConnected, self).__init__()
        self.fc1 = nn.Linear(latent_size, 64)
        self.fc2 = nn.Linear(64, 256)
        self.fc3 = nn.Linear(256, 28 * 28)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        x = x.reshape((-1, 1, 28, 28))
        return x
