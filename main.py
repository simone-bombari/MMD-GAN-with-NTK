import torch
import torch.optim as optim
from dataloader import load_data
from losses import mmd
from models import FullyConnected


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
lr = 0.5
weight_decay = 10 ** (-4)
dataset = 'MNIST'
batch_size = 512
noise_batch_size = 512
sigma = 6.5
latent_size = 32
train_loader, train_loader_with_replacement, test_loader, labels, num_classes = load_data(dataset, batch_size, download=False)
epochs = 2
net = FullyConnected(latent_size=latent_size)
net.to(device)
optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
save_path = './training4.pth'


for epoch in range(1, epochs + 1):
    print('epoch', epoch, flush=True)
    noise = torch.randn((noise_batch_size, latent_size)).to(device)
    c = 1
    # Train!
    for input_images, _ in iter(train_loader):
        if c % 10 == 0:
            print('minibatch', c, 'Loss = {:.3f}'.format(loss.item()), flush=True)  # MNIST has 60000 images
        generated_images = net(noise)
        input_images = input_images.to(device)
        # generated_images.to(device)
        optimizer.zero_grad()   # zero the gradient buffers
        loss = mmd(input_images, generated_images, sigma)  # Removing the dummy channel dim in MNIST
        # print('two parameters ', net.fc1.weight[0][0].item(), net.fc3.weight[0][0].item(), flush=True)
        loss.backward()
        optimizer.step()
        c += 1
    print('loss = ', loss.item(), flush=True)


torch.save(net.state_dict(), save_path)
