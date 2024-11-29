from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
from model import Net, get_device
import multiprocessing
from torch.optim.lr_scheduler import OneCycleLR


#### Model ####

device = get_device()
model = Net().to(device)

#### Dataloader ####
    
torch.manual_seed(1)
batch_size_train = 1024
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
        transform=train_transforms  # Applied to ALL training samples
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

def train(model, device, train_loader, optimizer, epoch, scheduler):
    model.train()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    pbar = tqdm(train_loader)
    correct = 0
    processed = 0
    
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Calculate training accuracy
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)
        
        pbar.set_description(desc= f'Loss={loss.item():.4f} Batch={batch_idx} Accuracy={100*correct/processed:0.2f}% LR={scheduler.get_last_lr()[0]:.6f}')


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
    
    # Initialize model and optimizer
    model = Net().to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.001,  # Set initial learning rate
        weight_decay=0.01  # Default AdamW weight decay
    )

    # Calculate number of steps for scheduler
    epochs = 20
    steps_per_epoch = len(train_loader)
    total_steps = epochs * steps_per_epoch

    # Initialize the scheduler
    scheduler = OneCycleLR(
        optimizer,
        max_lr=0.01,  # Maximum learning rate at peak
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.3,  # Percentage of training to increase LR
        div_factor=10,  # Initial LR = max_lr/div_factor
        final_div_factor=100,  # Final LR = initial_lr/final_div_factor
        anneal_strategy='cos'  # Cosine annealing
    )

    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch, scheduler)
        test(model, device, test_loader)
        # Print current learning rate
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")