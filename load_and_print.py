import torch
import matplotlib.pyplot as plt
from dataloader import load_data
from losses import mmd_gaussian
from models import FullyConnected


device = "cpu"
dataset = 'MNIST'

batch_size = 512
noise_batch_size = 512
train_loader, test_loader, labels, num_classes = load_data(dataset, batch_size, download=False)

sigma = 5
latent_size = 16

net = FullyConnected(latent_size=latent_size)
net.to(device)

net.load_state_dict(torch.load('./trained_classifier.pth', map_location=torch.device(device)))
net.eval()
check = False


if check:
    check_noise = torch.randn((noise_batch_size, latent_size)).float().to(device)
    check_input_images = next(iter(train_loader))[0]
    check_generated_images = net(check_noise)
    check_loss = mmd_gaussian(check_input_images, check_generated_images, sigma)
    print('check loss = ', check_loss.item(), flush=True)


with torch.no_grad():
    noise_to_print = torch.randn((16, latent_size))
    sample_images = net(noise_to_print)
    plot_image = sample_images.reshape((4, 4, 28, 28)).transpose(1, 2).reshape((-1, 4 * 28))
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(8, 8)
    ax.imshow(plot_image)
    fig.show()
    fig.savefig('MMD-GAN_digits.pdf')
