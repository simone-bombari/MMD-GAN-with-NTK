import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

import numpy as np


def gaussian_kernel(batch_images_1, batch_images_2, equal_batches=False):
    n_images_1, n_images_2 = batch_images_1.shape[0], batch_images_2.shape[0]
    # batch_images_1 = torch.view(n_images_1, -1)
    # batch_images_2 = torch.view(n_images_2, -1)  # to make the images flat
    dimension_images = batch_images_1.shape[1]  # for MNIST is 28 * 28
    term_for_mmd = 0
    print(n_images_1)
    for i in range(n_images_1):
        for j in range(n_images_2):
            if equal_batches:
                if i != j:
                    kernel_i_j = torch.exp(- (batch_images_1[i] - batch_images_1[j]).pow(2).sum())
                    term_for_mmd += kernel_i_j
            else:
                kernel_i_j = torch.exp(- (batch_images_1[i] - batch_images_1[j]).pow(2).sum())
                term_for_mmd += kernel_i_j
    if equal_batches:
        term_for_mmd /= n_images_1 * (n_images_1 - 1)
    else:
        term_for_mmd *= 2 / (n_images_1 * (n_images_1 - 1))
    return term_for_mmd


def mmd(batch_images, batch_generated_images):
    return gaussian_kernel(batch_images, batch_images, equal_batches=True) +\
           gaussian_kernel(batch_generated_images, batch_generated_images, equal_batches=True) -\
           gaussian_kernel(batch_images, batch_generated_images)  # There is definitely a smarter way to do this...


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
