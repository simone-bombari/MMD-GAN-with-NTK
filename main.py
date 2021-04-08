import torch
import torch.optim as optim
from dataloader import load_data
from losses import mmd
from models import FullyConnected


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset = 'MNIST'

lr = 0.2
weight_decay = 10 ** (-4)

batch_size = 512
noise_batch_size = 512
train_loader, test_loader, labels, num_classes = load_data(dataset, batch_size, download=False)

sigma = 5

latent_size = 8
net = FullyConnected(latent_size=latent_size)
net.to(device)

optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

epochs = 7

save_path = './training4.pth'


for epoch in range(1, epochs + 1):
    print('epoch', epoch, flush=True)
    noise = torch.randn((noise_batch_size, latent_size)).to(device)

    for c, (input_images, _) in enumerate(train_loader):
        generated_images = net(noise)
        input_images = input_images.to(device)

        optimizer.zero_grad()
        loss = mmd(input_images, generated_images, sigma)
        loss.backward()
        optimizer.step()

        if c % 10 == 9:
            print('minibatch', c+1, 'Loss = {:.3f}'.format(loss.item()), flush=True)

    scheduler.step()


torch.save(net.state_dict(), save_path)
