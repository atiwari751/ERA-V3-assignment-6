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
        self.bn2 = nn.BatchNorm2d(18)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(0.15)
        self.conv3 = nn.Conv2d(18, 18, 3)
        self.bn3 = nn.BatchNorm2d(18)
        self.conv4 = nn.Conv2d(18, 18, 3)
        self.bn4 = nn.BatchNorm2d(18)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout2d(0.15)
        self.conv5 = nn.Conv2d(18, 18, 3, padding=1)
        self.conv6 = nn.Conv2d(18, 18, 3, padding=1)
        #self.conv7 = nn.Conv2d(1024, 10, 3)
        self.fc1 = nn.Linear(4*4*18, 32)
        self.out = nn.Linear(32, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout1(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool2(F.relu(self.bn4(self.conv4(x))))
        x = self.dropout2(x)
        x = F.relu(self.conv5(x))
        #x = F.relu(self.conv6(x))
        x = x.reshape (-1, 4*4*18)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.out(x)
        return x
        #x = self.pool1(F.relu(self.conv2(F.relu(self.conv1(x)))))
        #x = self.pool2(F.relu(self.conv4(F.relu(self.conv3(x)))))
        #x = F.relu(self.conv6(F.relu(self.conv5(x))))
        #x = F.relu(self.conv7(x))
        #x = x.view(-1, 10)
        #return F.log_softmax(x)

def get_device():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    return device

if __name__ == '__main__':
    device = get_device()
    model = Net().to(device)
    summary(model, input_size=(1, 28, 28))