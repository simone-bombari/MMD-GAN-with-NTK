import torch
import torch.optim as optim
from dataloader import load_data
from losses import loss_calculator, compute_loss_accuracy
from models import Convolutional, FullyConnectedClassifier


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset = 'MNIST'

lr = 0.2
weight_decay = 10 ** (-4)

batch_size = 128
train_loader, test_loader, labels, num_classes = load_data(dataset, batch_size, download=False)

net = FullyConnectedClassifier()
net.to(device)

torch.save(net.state_dict(), './untrained_classifier.pth')

optimizer = optim.SGD(net.parameters(), lr=lr, weight_decay=weight_decay)  # Here SGD works better
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

epochs = 5
loss_function = 'CEL'

save_path = './trained_classifier_fc.pth'


for epoch in range(1, epochs + 1):
    print('Epoch', epoch, flush=True)

    for c, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_calculator(outputs, labels, loss_function)
        loss.backward()
        optimizer.step()

        if (c+1) % 100 == 0:
            print('minibatch', c+1, 'Loss = {:.3f}'.format(loss.item()), flush=True)

    scheduler.step()
    # Evaluate!
    with torch.no_grad():

        train_loss, train_accuracy = compute_loss_accuracy(train_loader, loss_function, net, device)
        test_loss, test_accuracy = compute_loss_accuracy(test_loader, loss_function, net, device)

        print('Overall:\nTrain loss = {:.3f}\tTrain accuracy = {:.3f}\n'
              'Test loss = {:.3f}\tTest accuracy = {:.3f}\n'.format(
               train_loss, train_accuracy, test_loss, test_accuracy))


torch.save(net.state_dict(), save_path)
