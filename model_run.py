from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
from model import Net, get_device
import multiprocessing


#### Model ####

device = get_device()
model = Net().to(device)

#### Dataloader ####
    
torch.manual_seed(1)
batch_size_train = 5000
batch_size_test = 1000

kwargs = {'num_workers': 4, 'pin_memory': True} if device.type == "cuda" else {}

# Define augmentation for training data
train_transforms = transforms.Compose([
    # Spatial augmentations
    transforms.RandomRotation((-10, 10)),              # Random rotation ±8 degrees
    transforms.RandomAffine(
        degrees=0,                                   # No additional rotation
        translate=(0.1, 0.1),                        # Random shift up to 10%
        scale=(0.9, 1.1),                           # Random scaling ±10%
        shear=(-5, 5)
        #fillcolor=0                                  # Fill empty space with black
    ),
    
    # Convert to tensor and normalize
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Keep test transforms simple
test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Create data loaders with respective transforms
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        '../data', 
        train=True, 
        download=True,
        transform=train_transforms  # Apply augmentation only to training data
    ),
    batch_size=batch_size_train, 
    shuffle=True, 
    **kwargs
)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        '../data', 
        train=False, 
        transform=test_transforms  # No augmentation for test data
    ),
    batch_size=batch_size_test, 
    shuffle=True, 
    **kwargs
)

#### Train and run ####

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    pbar = tqdm(train_loader)
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        #loss = F.nll_loss(output, target)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        pbar.set_description(desc= f'loss={loss.item()} batch_id={batch_idx}')


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


if __name__ == '__main__':
    multiprocessing.freeze_support()
    
    optimizer = optim.AdamW(model.parameters(), lr=0.009, weight_decay=0.00001)

    for epoch in range(1, 21):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)