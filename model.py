from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchsummary import summary

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 3) #input -? OUtput? RF
        self.bn1 = nn.BatchNorm2d(10)
        self.conv2 = nn.Conv2d(10, 18, 3)
        #self.bn2 = nn.BatchNorm2d(18)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(0.15)
        self.conv3 = nn.Conv2d(18, 18, 3)
        #self.bn3 = nn.BatchNorm2d(18)
        self.conv4 = nn.Conv2d(18, 18, 3)
        self.bn4 = nn.BatchNorm2d(18)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout2d(0.15)
        self.conv5 = nn.Conv2d(18, 18, 3, padding=1)
        #self.conv6 = nn.Conv2d(18, 18, 3, padding=1)
        #self.conv7 = nn.Conv2d(1024, 10, 3)
        self.fc1 = nn.Linear(4*4*18, 30)
        self.out = nn.Linear(30, 10)

    def forward(self, x):
        # First Convolution Block
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        # Second Convolution Block
        x = self.conv2(x)
        #x = self.bn2(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        # Third Convolution Block
        x = self.conv3(x)
        #x = self.bn3(x)
        x = F.relu(x)

        # Fourth Convolution Block
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        # Fifth Convolution Block
        x = self.conv5(x)
        x = F.relu(x)

        # Sixth Convolution Block (commented out in your code)
        # x = self.conv6(x)
        # x = F.relu(x)

        # Flatten and Dense Layers
        x = x.reshape(-1, 4*4*18)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.out(x)
        
        return x

def get_device():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    return device

if __name__ == '__main__':
    device = get_device()
    model = Net().to(device)
    summary(model, input_size=(1, 28, 28))