import some_code
import some_more_code
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import pickle
import torch.optim as optim
import numpy as np
from some_code import load_data, loss_calculator, compute_loss_accuracy
from some_more_code import mmd_generator, mmd

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
lr = 0.1
loss_function = 'CEL'
dataset = 'MNIST'
batch_size = 128
noise_batch_size = 50
latent_size = 20
train_loader, train_loader_with_replacement, test_loader, labels, num_classes = load_data(dataset, batch_size, download=False)
epochs = 5
net = mmd_generator(latent_size=latent_size)
net.to(device)
optimizer = optim.SGD(net.parameters(), lr=lr)
save_path = './prova_net.pth'


for epoch in range(epochs):

    noise = torch.randn((noise_batch_size, latent_size))
    generated_images = net(noise)

    # Train!
    for input_images, labels in iter(train_loader):
        input_images = input_images.to(device)
        # labels = labels.to(device)
        optimizer.zero_grad()   # zero the gradient buffers
        loss = mmd(input_images.squeeze(), generated_images.squeeze())
        loss.backward()
        optimizer.step()
    # print(total_labels)


torch.save(net.state_dict(), save_path)

with torch.no_grad():
    noise_to_print = torch.randn((1, latent_size))
    sample_image = net(noise_to_print)
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(2, 2)
    sample_image = sample_image.squeeze()
    ax.imshow(sample_image)
    plt.show()
