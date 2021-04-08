import torch
import matplotlib.pyplot as plt
from dataloader import load_data
from losses import mmd
from models import mmd_generator


device = "cpu"
lr = 0.5
weight_decay = 10 ** (-4)
dataset = 'MNIST'
batch_size = 512
noise_batch_size = 512
train_loader, train_loader_with_replacement, test_loader, labels, num_classes = load_data(dataset, batch_size,
                                                                                          download=False)
sigma = 6.5
latent_size = 50
net = mmd_generator(latent_size=latent_size)
net.to(device)

net.load_state_dict(torch.load('./saved_models/training4_1epoch.pth', map_location=torch.device('cpu')))
net.eval()


with torch.no_grad():

    check_noise = torch.randn((noise_batch_size, latent_size)).float().to(device)
    check_input_images = next(iter(train_loader))[0]
    check_generated_images = net(check_noise)
    check_loss = mmd(check_input_images.squeeze(), check_generated_images.squeeze(), sigma)
    print('check loss = ', check_loss.item(), flush=True)

    noise_to_print = torch.randn((1, latent_size))
    sample_image = net(noise_to_print)
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(2, 2)
    sample_image = sample_image.squeeze()
    ax.imshow(sample_image, cmap='gray')
    fig.show()
