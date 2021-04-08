import torch
import torchvision
import pickle


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

    test_loader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=batch_size
    )

    return train_loader, test_loader, labels, num_classes


