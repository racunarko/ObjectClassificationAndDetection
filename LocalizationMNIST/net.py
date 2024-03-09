import torch.nn as nn
import torch.nn.functional as F
from transformations import GaussianNoise

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1, stride=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.5)
        
        # classification layer
        self.fc2 = nn.Linear(256, 10)
        
        # bbox regression layer   
        self.x1_out = nn.Linear(256, 1)
        self.y1_out = nn.Linear(256, 1)
        self.x2_out = nn.Linear(256, 1)
        self.y2_out = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.pool1(self.conv1(x))))
        x = F.relu(self.bn2(self.pool2(self.conv2(x))))
        x = F.relu(self.bn3(self.pool3(self.conv3(x))))

        x = x.view(-1, 128 * 8 * 8) # flatten
        x = F.relu(self.bn4(self.fc1(x)))
        x = self.dropout1(x) # applyjam dropout nakon aktivacije potpuno povezanog sloja
        
        # classification
        clsf = F.log_softmax(self.fc2(x), dim=1)
        
        # bbox regression
        x1 = self.x1_out(x)
        y1 = self.y1_out(x)
        x2 = self.x2_out(x)
        y2 = self.y2_out(x)
        
        return clsf, x1.squeeze(), y1.squeeze(), x2.squeeze(), y2.squeeze()