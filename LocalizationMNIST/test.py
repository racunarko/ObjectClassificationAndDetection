import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from dataset import MNISTLocalizationDataset
from net import Net

import matplotlib.pyplot as plt
import numpy as np
import cv2

device = 'mps'
image_size = 64
batch_size_train = 64
batch_size_test = 64
transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])



train_set = MNISTLocalizationDataset(image_size, transform, train_set=True)
test_set = MNISTLocalizationDataset(image_size, transform, train_set=False)
    
train_loader = DataLoader(train_set, batch_size=batch_size_train, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size_test, shuffle=True)
    
network = Net().to('mps')
examples = enumerate(test_loader)
batch_idx, (example_data, example_targets, example_bbox) = next(examples)
    
network.load_state_dict(torch.load('mnist_localization_randomsizes.pth'))
network.eval()
    
with torch.no_grad():
    clsf_out, x1_out, y1_out, x2_out, y2_out = network(example_data.to(device))

    plt.figure(figsize=(32, 32))
    for idx in range(0, example_data.shape[0]):
        image = np.array(example_data[idx, 0, ...]).copy()

        x1, y1, x2, y2 = list(map(lambda x: int(x.item()), [x1_out[idx, ...], y1_out[idx, ...], x2_out[idx, ...], y2_out[idx, ...]]))
        image = cv2.rectangle(image, (x1, y1), (x2, y2), (2.5), 2)

        plt.subplot(12, 12, idx+1)
        plt.imshow(image, cmap='gray')
        plt.title(np.argmax(clsf_out[idx, ...].cpu()))
        plt.xticks([])
        plt.yticks([])

    plt.savefig('tested_examples.png')
    plt.show()