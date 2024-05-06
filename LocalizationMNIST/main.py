import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, Dataset, DataLoader

from torchsummary import summary

import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from dataset import MNISTLocalizationDataset
from net import Net
from transformations import GaussianBlur
from transformations import GaussianNoise

epochs = 20
learning_rate = 0.01
log_interval = 100
image_size = 64

def train_step(network, optimizer, train_loader, epoch, device, verbose=True):
    train_losses = []
    train_counter = []

    network.train()

    for batch_idx, (data, target_class, target_bbox) in enumerate(train_loader):
        data = data.to(device)
        target_class = target_class.to(device)
        target_bbox = [x.float().to(device) for x in target_bbox]
        
        target_x1, target_y1, target_x2, target_y2 = target_bbox[0] / image_size, target_bbox[1] / image_size, target_bbox[2] / image_size, target_bbox[3] / image_size
        
        # -------
        # 1. Korak racunam prolaz unaprijed
        output = network(data)
        
        pred_class = output[0]
        pred_x1, pred_y1, pred_x2, pred_y2 = output[1] / image_size, output[2] / image_size, output[3]/ image_size, output[4] / image_size
        
        # 2. Korak racunam negative log likelihood loss i MSE loss za bbox koordinate
        loss_clsf = F.nll_loss(pred_class, target_class)
        loss_bbox = F.mse_loss(pred_x1, target_x1) + F.mse_loss(pred_y1, target_y1) + F.mse_loss(pred_x2, target_x2) + F.mse_loss(pred_y2, target_y2)
        loss = loss_clsf + loss_bbox
        
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

    test_loss_clsf = 0
    test_loss_bbox = 0
    correct = 0
    correct_bbox = 0
    with torch.no_grad():
        for data, target_class, target_bbox in test_loader:
            data = data.to(device)
            target_class = target_class.to(device)
            target_bbox = [x.float().to(device) for x in target_bbox]
            
            target_x1, target_y1, target_x2, target_y2 = target_bbox[0] / image_size, target_bbox[1] / image_size, target_bbox[2] / image_size, target_bbox[3] / image_size
            
            output = network(data)
            pred_class = output[0]
            pred_x1, pred_y1, pred_x2, pred_y2 = output[1] / image_size, output[2] / image_size, output[3]/ image_size, output[4] / image_size

            # compute classification loss
            test_loss_clsf += F.nll_loss(pred_class, target_class, reduction='sum').item()
            
            # compute bbox regression loss
            test_loss_bbox += F.mse_loss(pred_x1, target_x1, reduction='sum').item()
            test_loss_bbox += F.mse_loss(pred_y1, target_y1, reduction='sum').item()
            test_loss_bbox += F.mse_loss(pred_x2, target_x2, reduction='sum').item()
            test_loss_bbox += F.mse_loss(pred_y2, target_y2, reduction='sum').item()
            
            # compute accuracy
            pred = pred_class.argmax(dim = 1, keepdim=True)
            correct += pred.eq(target_class.view_as(pred)).sum().item()
            
            for idx in range(data.size(0)):
                pred_bbox = [pred_x1[idx], pred_y1[idx], pred_x2[idx], pred_y2[idx]]
                target_bbox = [target_x1[idx], target_y1[idx], target_x2[idx], target_y2[idx]]
                
                pred_bbox = torch.tensor(pred_bbox)
                target_bbox = torch.tensor(target_bbox)
                iou = intersection_over_union(pred_bbox.cpu().numpy(), target_bbox.cpu().numpy(), 0.5)
                if iou:
                    correct_bbox += 1
            
    test_loss_clsf /= len(test_loader.dataset)
    test_loss_bbox /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    test_accuracy_bbox = 100. * correct_bbox / len(test_loader.dataset)

    if verbose:
        print('\n[Test] Classification: Avg. loss: {:.4f}, Accuracy: {:5d}/{:5d} ({:2.2f}%) | Object detection: Avg. loss: {:.4f}, Accuracy: {:5d}/{:5d} ({:2.2f}%)\n'.format(
            test_loss_clsf,
            correct,
            len(test_loader.dataset),
            test_accuracy,
            test_loss_bbox,
            correct_bbox,
            len(test_loader.dataset),
            test_accuracy_bbox))

    return test_loss_clsf, test_accuracy, correct, test_loss_bbox

def train_network(network, optimizer, train_loader, test_loader, device='cpu'):
    train_losses = []
    train_counter = []
    test_losses_clsf = []
    test_losses_bbox = []
    test_counter = [i*len(train_loader.dataset) for i in range(epochs + 1)]
    test_accuracy = []
    # ----- testiranje slucajnog modela
    test_loss_clsf, accuracy, correct, test_loss_bbox = test(network, test_loader, device)
    test_losses_clsf.append(test_loss_clsf)
    test_losses_bbox.append(test_loss_bbox)
    test_accuracy.append(accuracy)
    # -------------

    for epoch in range(1, epochs + 1):
        # --------- treniranje modela
        new_train_losses, new_train_counter = train_step(network, optimizer, train_loader, epoch, device)
        train_losses.extend(new_train_losses)
        train_counter.extend(new_train_counter)
        
        test_loss_clsf, accuracy, correct, test_loss_bbox = test(network, test_loader, device)
        test_losses_clsf.append(test_loss_clsf)
        test_accuracy.append(accuracy)
        test_losses_bbox.append(test_loss_bbox)
        # ---------

    return train_losses, train_counter, test_losses_clsf, test_accuracy, test_losses_bbox, test_counter

def get_number_of_model_parameters(model):
    return sum(p.numel() for p in model.parameters())

def intersection_over_union(pred, target, threshold):
    x1 = max(pred[0], target[0])
    y1 = max(pred[1], target[1])
    x2 = min(pred[2], target[2])
    y2 = min(pred[3], target[3])
    
    if x2 < x1 or y2 < y1:
        return False
    
    intersection = (x2 - x1) * (y2 - y1)
    pred_area = (pred[2] - pred[0]) * (pred[3] - pred[1])
    target_area = (target[2] - target[0]) * (target[3] - target[1])
    
    iou = intersection / float(pred_area + target_area - intersection)
    return iou > threshold
    
    
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
    
    # radim jos na transformacijama, i trebalo bi istrenirati 20 epoha na kompu na GPU
    train_trainsform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation(3),
        # GaussianBlur(kernel=3, sigma=(0.1, 0.2)),
        # GaussianNoise(mean=0., var=0.05, clip=True),
        # ovo bas nije pomoglo, dodao bih jedino jos gaussian blur i mozda neki cutout
        
        # transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_set = MNISTLocalizationDataset(image_size, train_trainsform, train_set=True)
    test_set = MNISTLocalizationDataset(image_size, transform, train_set=False)
    
    train_loader = DataLoader(train_set, batch_size=batch_size_train, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size_test, shuffle=True)
    
    network = Net().to(device)
    
    # Adam ima puno bolji rezultat na mps-u, nego SGD
    # probaj SGD na GPU-u
    optimizer = optim.Adam(network.parameters(), lr=learning_rate)
    
    # ovo ako radis na 'cpu' ili gpu
    # summary(network, (1, 64, 64), device=device)
    train_losses, train_counter, test_losses_clsf, test_accuracy, test_losses_bbox, test_counter = train_network(network, optimizer, train_loader, test_loader, device)
    
    torch.save(network.state_dict(), './mnist_localization_randomsizes.pth')
    
    # visualize_predictions(network, device, test_loader, num_images=10)
    
    # ovo jos nisam siguran kako da napravim, mozda bolje koristiti predict.py a da ovo samo bude za gradnju modela i treniranje
    # isto mogu jos napraviti nekakvu evaluaciju matematicku ali to isto mogu u posebnu datotektu staviti 
    
    # # predict on model
    # image = cv2.imread('sedmica.png', cv2.IMREAD_GRAYSCALE)
    # image = cv2.resize(image, (28, 28))
    # mean = 0.1307
    # std = 0.3081
    # image = image / 255.

    # image = (image - mean) / std
    # image = torch.tensor(image).float().unsqueeze(0).unsqueeze(0)
    
    # network.eval()
    # with torch.no_grad():
    #     output = network(image.to(device))
    #     pred = output.argmax(dim=1, keepdim=True)
    #     print(f"Predicted: {pred.item()}")        
    
if __name__ == "__main__":
    main()
