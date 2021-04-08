import torch
from torch import nn


def gaussian_kernel(img_1, img_2, sigma):
    return torch.exp(- (img_1 - img_2).pow(2).sum() / (2 * sigma ** 2))


def mmd_equaterm(batch, sigma):
    n_images = batch.shape[0]
    dist = torch.cdist(batch.flatten(start_dim=2).transpose(0, 1),
                       batch.flatten(start_dim=2).transpose(0, 1))
    diag = torch.diag(dist.squeeze())
    gauss_dist = torch.exp(- dist.pow(2) / (2 * sigma ** 2)).sum() \
                 - torch.exp(- diag.pow(2) / (2 * sigma ** 2)).sum()
    return gauss_dist / (n_images * (n_images - 1))


def mmd_crossterm(batch_images, batch_generated_images, sigma):
    n_images_1, n_images_2 = batch_images.shape[0], batch_generated_images.shape[0]
    dist = torch.cdist(batch_images.flatten(start_dim=2).transpose(0, 1),
                       batch_generated_images.flatten(start_dim=2).transpose(0, 1))
    gauss_dist = torch.exp(- dist.pow(2) / (2 * sigma ** 2)).sum()
    return -2 * gauss_dist / (n_images_1 * n_images_2)


def mmd(batch_images, batch_generated_images, sigma):
    first_term = mmd_equaterm(batch_images, sigma)
    second_term = mmd_equaterm(batch_generated_images, sigma)
    third_term = mmd_crossterm(batch_images, batch_generated_images, sigma)
    return first_term + second_term + third_term


def loss_calculator(outputs, labels, loss_function, num_classes=10):
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
