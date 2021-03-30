import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import pickle
import torch.optim as optim
import numpy as np


def load_data(dataset, batch_size, download=False):

    if dataset == 'MNIST':
        dataset_train = torchvision.datasets.MNIST('./datasets/',
                                                   train=True,
                                                   transform=torchvision.transforms.ToTensor(),
                                                   download=download)
        dataset_test = torchvision.datasets.MNIST('./datasets/',
                                                  train=False,
                                                  transform=torchvision.transforms.ToTensor(),
                                                  download=download)
        labels = list(range(10))
        num_classes = 10

    elif dataset == 'CIFAR10':
        dataset_train = torchvision.datasets.CIFAR10('./datasets/',
                                                     train=True,
                                                     transform=torchvision.transforms.ToTensor(),
                                                     download=download)
        dataset_test = torchvision.datasets.CIFAR10('./datasets/',
                                                    train=False,
                                                    transform=torchvision.transforms.ToTensor(),
                                                    download=download)
        with open('./datasets/cifar-10-batches-py/batches.meta', 'rb') as f:
            dict = pickle.load(f)
            labels = dict['label_names']
        num_classes = 10

    elif dataset == 'CIFAR100':
        dataset_train = torchvision.datasets.CIFAR10('./datasets/',
                                                     train=True,
                                                     transform=torchvision.transforms.ToTensor(),
                                                     download=download)
        dataset_test = torchvision.datasets.CIFAR10('./datasets/',
                                                    train=False,
                                                    transform=torchvision.transforms.ToTensor(),
                                                    download=download)
        with open('./datasets/cifar-100-python/meta', 'rb') as f:
            dict = pickle.load(f)
            labels = dict['fine_label_names']
        num_classes = 100

    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True
        )

    train_loader_with_replacement = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        sampler=torch.utils.data.sampler.RandomSampler(dataset_train, replacement=True)
    )

    test_loader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=batch_size
    )

    return train_loader, train_loader_with_replacement, test_loader, labels, num_classes


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


def loss_calculator(outputs, labels, loss_function, num_classes):
    if loss_function == 'MSE':
        criterion = nn.MSELoss()
        targets = torch.eye(num_classes)[labels]  # One-hot encoding
    elif loss_function == 'CEL':
        criterion = nn.CrossEntropyLoss()
        targets = labels

    return criterion(outputs, targets)


def compute_loss_accuracy(data_loader, loss_function, net, device):
    score = 0
    samples = 0
    full_loss = 0
    for input_images, labels in iter(data_loader):
        input_images = input_images.to(device)
        labels = labels.to(device)
        outputs = net(input_images)
        minibatch_loss = loss_calculator(outputs, labels, loss_function).item()
        predicted = torch.max(outputs, 1)[1]  # Max on the first axis, in 0 we have the value of the max.

        minibatch_score = (predicted == labels).sum().item()
        minibatch_size = len(labels)  # Can be different in the last iteration
        score += minibatch_score
        full_loss += minibatch_loss * minibatch_size
        samples += minibatch_size

    loss = full_loss / samples
    accuracy = score / samples

    return loss, accuracy
