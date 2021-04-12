import torch
from torch import nn

'''

Useful stuff for computing the MMD estimator with a Gaussian kernel

'''

def mmd_equaterm_gaussian(batch, sigma):
    n_images = batch.shape[0]
    dist = torch.cdist(batch.flatten(start_dim=2).transpose(0, 1),
                       batch.flatten(start_dim=2).transpose(0, 1))
    diag = torch.diag(dist.squeeze())
    gauss_dist = torch.exp(- dist.pow(2) / (2 * sigma ** 2)).sum() \
                 - torch.exp(- diag.pow(2) / (2 * sigma ** 2)).sum()
    return gauss_dist / (n_images * (n_images - 1))


def mmd_crossterm_gaussian(batch_images, batch_generated_images, sigma):
    n_images_1, n_images_2 = batch_images.shape[0], batch_generated_images.shape[0]
    dist = torch.cdist(batch_images.flatten(start_dim=2).transpose(0, 1),
                       batch_generated_images.flatten(start_dim=2).transpose(0, 1))
    gauss_dist = torch.exp(- dist.pow(2) / (2 * sigma ** 2)).sum()
    return -2 * gauss_dist / (n_images_1 * n_images_2)


def mmd_gaussian(batch_images, batch_generated_images, sigma):
    first_term = mmd_equaterm_gaussian(batch_images, sigma)
    second_term = mmd_equaterm_gaussian(batch_generated_images, sigma)
    third_term = mmd_crossterm_gaussian(batch_images, batch_generated_images, sigma)
    return first_term + second_term + third_term

# ----------------------------------------------------------------------------------------------------------------------

'''

Useful stuff for computing the MMD estimator with a Gaussian kernel

'''

'''def mmd_equaterm_NTK(batch_images, net):


def mmd_crossterm_NTK(batch_images, batch_generated_images, net):
'''


def get_NTK_avg_feature(images, classifier, retain_graph=False):
    outputs = classifier(images)
    avg_outputs = outputs.mean(dim=0)
    squared_outputs = (avg_outputs ** 2).sum()
    # squared_outputs.backward(retain_graph=True)
    parameters_grad = []
    for param in iter(classifier.parameters()):
        # parameters_grad.append(param.grad)
        parameters_grad.append(torch.autograd.grad(squared_outputs, param,
                                                   retain_graph=True, create_graph=True)[0])
    return parameters_grad


def distance_features(list1, list2):
    dist = 0
    l = len(list1)
    for i in range(l):
        tensor_diff = list1[i] - list2[i]
        dist += (tensor_diff ** 2).sum()
    return dist

#  Calling backward() twice would just add the gradients

def mmd_NTK(batch_images, batch_generated_images, classifier):
    avg_feature_vector_images = get_NTK_avg_feature(batch_images, classifier)
    # print(avg_feature_vector_images[0][0])
    avg_feature_vector_generated_images = get_NTK_avg_feature(batch_generated_images, classifier)
    # print(avg_feature_vector_images[0][0])
    # print(avg_feature_vector_generated_images[0][0])
    distance_squared = distance_features(avg_feature_vector_images, avg_feature_vector_generated_images)
    return distance_squared


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
