import torch
import torch.optim as optim
from dataloader import load_data
from losses import mmd_gaussian, mmd_NTK
from models import FullyConnected, Convolutional


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset = 'MNIST'
kernel = 'NTK'

lr = 0.1
weight_decay = 10 ** (-4)

batch_size = 512
noise_batch_size = 512
train_loader, test_loader, labels, num_classes = load_data(dataset, batch_size, download=False)

latent_size = 16
net = FullyConnected(latent_size=latent_size)
net.to(device)

optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.7)

epochs = 800

save_path = './training_NTK.pth'

if kernel == 'NTK':
    classifier = Convolutional()
    classifier.to(device)
    classifier.load_state_dict(torch.load('./trained_classifier.pth', map_location=torch.device(device)))
elif kernel == 'Gaussian':
    sigma = 5

for epoch in range(1, epochs + 1):
    print('epoch', epoch, flush=True)
    noise = torch.randn((noise_batch_size, latent_size)).to(device)

    for c, (input_images, _) in enumerate(train_loader):
        generated_images = net(noise)
        input_images = input_images.to(device)

        optimizer.zero_grad()
        # loss = mmd_gaussian(input_images, generated_images, sigma)
        loss = mmd_NTK(input_images, generated_images, classifier)
        # loss.retain_grad()
        loss.backward()
        '''print(net.fc1.weight[0][0:3])
        print(net.fc2.weight[0][0:3])
        print(net.fc3.weight[0][0:3])'''
        optimizer.step()

        if c % 10 == 9:
            print('minibatch', c+1, 'Loss = {:.3f}'.format(loss.item()), flush=True)
            # print(classifier.fc1.weight[0][0].item())

    scheduler.step()


torch.save(net.state_dict(), save_path)
