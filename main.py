import torch
import matplotlib.pyplot as plt
import torch.optim as optim
from dataloader import load_data
from losses import mmd
from models import mmd_generator
from torch import nn


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
lr = 0.5
weight_decay = 10 ** (-4)
dataset = 'MNIST'
batch_size = 512
noise_batch_size = 512
sigma = 50
latent_size = 50
train_loader, train_loader_with_replacement, test_loader, labels, num_classes = load_data(dataset, batch_size, download=False)
epochs = 5
net = mmd_generator(latent_size=latent_size)
net.to(device)
optimizer = optim.SGD(net.parameters(), lr=lr, weight_decay=weight_decay)
save_path = './training2.pth'


for epoch in range(epochs):
    print('epoch', epoch, flush=True)
    noise = torch.randn((noise_batch_size, latent_size)).float().to(device)
    print(device, flush=True)
    print(next(net.parameters()).device, flush=True)
    print(noise.device, '\n', flush=True)
    c = 0
    # Train!
    for input_images, _ in iter(train_loader):
        if c == 110:
            break
        print('minibatch', c, flush=True)  # MNIST has 60000 images
        generated_images = net(noise)
        input_images = input_images.float().to(device)
        # generated_images.to(device)
        optimizer.zero_grad()   # zero the gradient buffers
        loss = mmd(input_images.squeeze(), generated_images.squeeze(), sigma)
        print('loss = ', loss.item(), flush=True)
        print('two parameters ', net.decoder[0].weight[0][0].item(), net.decoder[9].weight[0][0][0][0].item(), flush=True)
        loss.backward()
        optimizer.step()
        c += 1
    # print(total_labels)


torch.save(net.state_dict(), save_path)
 
'''
with torch.no_grad():
    noise_to_print = torch.randn((1, latent_size))
    sample_image = net(noise_to_print)
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(2, 2)
    sample_image = sample_image.squeeze()
    ax.imshow(sample_image)
    plt.show()
'''