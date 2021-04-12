import torch
import torch.optim as optim
from dataloader import load_data
from losses import mmd_gaussian
from models import FullyConnected, Convolutional


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset = 'MNIST'
kernel = 'Gaussian'

lr = 0.2
weight_decay = 10 ** (-4)

batch_size = 512
noise_batch_size = 512
train_loader, test_loader, labels, num_classes = load_data(dataset, batch_size, download=False)

sigma = 5

latent_size = 16
net = FullyConnected(latent_size=latent_size)
net.to(device)

optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

epochs = 20

save_path = './training4.pth'

if kernel == 'NTK':
    classifier = Convolutional()
    classifier.to(device)
    classifier.load_state_dict(torch.load('./trained_classifier.pth', map_location=torch.device(device)))


for epoch in range(1, epochs + 1):
    print('epoch', epoch, flush=True)
    noise = torch.randn((noise_batch_size, latent_size)).to(device)

    for c, (input_images, _) in enumerate(train_loader):
        generated_images = net(noise)
        input_images = input_images.to(device)

        optimizer.zero_grad()
        loss = mmd_gaussian(input_images, generated_images, sigma)
        # loss = mmd_NTK(input_images, generated_images, classifier)
        loss.retain_grad()
        loss.backward()
        optimizer.step()

        if c % 10 == 9:
            print('minibatch', c+1, 'Loss = {:.3f}'.format(loss.item()), flush=True)

    scheduler.step()


torch.save(net.state_dict(), save_path)
