import torch
from torch import nn


def gaussian_kernel(batch_images_1, batch_images_2, sigma, equal_batches=False):
    n_images_1, n_images_2 = batch_images_1.shape[0], batch_images_2.shape[0]
    term_for_mmd = 0
    for i in range(n_images_1):
        for j in range(n_images_2):
            if equal_batches:
                if i != j:
                    kernel_i_j = torch.exp(- (batch_images_1[i] - batch_images_2[j]).pow(2).sum() / (2 * sigma ** 2))
                    term_for_mmd += kernel_i_j
            else:
                kernel_i_j = torch.exp(- (batch_images_1[i] - batch_images_2[j]).pow(2).sum() / (2 * sigma ** 2))
                term_for_mmd += kernel_i_j
    if equal_batches:
        term_for_mmd /= n_images_1 * (n_images_1 - 1)
    else:
        term_for_mmd *= 2 / (n_images_1 * (n_images_1 - 1))
    return term_for_mmd


def mmd(batch_images, batch_generated_images, sigma):
    return gaussian_kernel(batch_images, batch_images, sigma, equal_batches=True) +\
           gaussian_kernel(batch_generated_images, batch_generated_images, sigma, equal_batches=True) -\
           gaussian_kernel(batch_images, batch_generated_images, sigma)  # There is definitely a smarter way to do this...


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
