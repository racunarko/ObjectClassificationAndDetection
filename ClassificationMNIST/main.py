import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, Dataset, DataLoader

from torchsummary import summary

import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from net import Net

epochs = 5
learning_rate = 0.01
log_interval = 100

def train_step(network, optimizer, train_loader, epoch, device, verbose=True):
    train_losses = []
    train_counter = []

    network.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)

        # -------
        # 1. Korak racunam prolaz unaprijed
        output = network(data)

        # 2. Korak racunam negative log likelihood loss
        loss = F.nll_loss(output, target)

        # 3. Korak propagacija greske unazad
        optimizer.zero_grad()
        loss.backward()

        # 4. Korak optimizacija
        optimizer.step()
        # -----------

        if (batch_idx % log_interval == 0):
            if verbose:
                print('Train Epoch: {:5d} [{:5d}/{:5d} ({:2.0f}%)]\tLoss: {:.6f}'.format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item()))

            train_losses.append(loss.item())
            train_counter.append((batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))

    return train_losses, train_counter

def test(network, test_loader, device, verbose=True):
    network.eval()

    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)

            # -----
            output = network(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim = 1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            # -------

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    if verbose:
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {:5d}/{:5d} ({:2.2f}%)\n'.format(
            test_loss,
            correct,
            len(test_loader.dataset),
            accuracy))

    return test_loss, accuracy

def train_network(network, optimizer, train_loader, test_loader, device='cpu'):
    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i*len(train_loader.dataset) for i in range(epochs + 1)]
    # ----- testiranje slucajnog modela
    test_accuracy = []
    test_loss, accuracy = test(network, test_loader, device)
    test_accuracy.append(accuracy)
    # -------------
    test_losses.append(test_loss)

    for epoch in range(1, epochs + 1):
        # --------- treniranje modela
        new_train_losses, new_train_counter = train_step(network, optimizer, train_loader, epoch, device)
        test_loss, accuracy = test(network, test_loader, device)
        test_accuracy.append(accuracy)
        # ---------

        train_losses.extend(new_train_losses)
        train_counter.extend(new_train_counter)
        test_losses.append(test_loss)

    return train_losses, train_counter, test_losses, test_counter, test_accuracy

def get_number_of_model_parameters(model):
    return sum(p.numel() for p in model.parameters())

def visualize_predictions(network, device, test_loader, num_images=10):
    network.eval()
    images, predictions, true_labels = [], [], []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = network(data)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

            images.extend(data.cpu().numpy())
            predictions.extend(pred.view_as(target).cpu().numpy())
            true_labels.extend(target.cpu().numpy())

            if len(images) >= num_images:
                break

    fig = plt.figure(figsize=(10, num_images * 2))
    for i in range(0, num_images):
        ax = fig.add_subplot(num_images, 1, i + 1, xticks=[], yticks=[])
        image = np.squeeze(images[i])
        ax.imshow(image, cmap="gray")
        ax.set_title(f"True: {true_labels[i]}, Predicted: {predictions[i]}")

    plt.show()

def visualize_predictions(network, device, test_loader, num_images=10):
    network.eval()
    images, predictions, true_labels = [], [], []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = network(data)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

            images.extend(data.cpu().numpy())
            predictions.extend(pred.view_as(target).cpu().numpy())
            true_labels.extend(target.cpu().numpy())

            if len(images) >= num_images:
                break

    fig = plt.figure(figsize=(10, num_images * 2))
    for i in range(0, num_images):
        ax = fig.add_subplot(num_images, 1, i + 1, xticks=[], yticks=[])
        image = np.squeeze(images[i])
        ax.imshow(image, cmap="gray")
        ax.set_title(f"True: {true_labels[i]}, Predicted: {predictions[i]}")

    plt.show()


def main():
    
    device = 'mps' 
    
    batch_size_train = 64
    batch_size_test = 64
    
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_set = torchvision.datasets.MNIST('./data/', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.MNIST('./data/', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_set, batch_size=batch_size_train, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size_test, shuffle=True)
    
    network = Net().to(device)
    optimizer = optim.SGD(network.parameters(), lr=learning_rate)
    
    train_losses, train_counter, test_losses, test_counter, test_accuracy = train_network(network, optimizer, train_loader, test_loader, device)
    
    torch.save(network.state_dict(), './mnist_classification.pth')
    
    visualize_predictions(network, device, test_loader, num_images=10)
    
    # predict on model
    image = cv2.imread('sedmica.png', cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (28, 28))
    mean = 0.1307
    std = 0.3081
    image = image / 255.

    image = (image - mean) / std
    image = torch.tensor(image).float().unsqueeze(0).unsqueeze(0)
    
    network.eval()
    with torch.no_grad():
        output = network(image.to(device))
        pred = output.argmax(dim=1, keepdim=True)
        print(f"Predicted: {pred.item()}")        
    
if __name__ == "__main__":
    main()
