import torch
import torchvision
import matplotlib.pyplot as plt

from torch import nn
from torchvision import transforms
from torchinfo import summary
from helper_functions import plot_images, set_seed

# Choose device 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'DEVICE: {device}')

# Set random seed
set_seed(seed=100)

# Custom parameters
SHOW_ORIGINAL_IMAGES = True

# Hyper-parameters
NUM_EPOCHS = 10
BATCH_SIZE = 1024
LEARNING_RATE = 0.001

# Image preprocessing modules
transform = transforms.Compose([
    transforms.Pad(4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor()])
print(f"Used transform: {transform}")

# CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(root='../data/',
                                             train=True, 
                                             transform=transform,
                                             download=True)

test_dataset = torchvision.datasets.CIFAR10(root='../data/',
                                            train=False, 
                                            transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=BATCH_SIZE,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=BATCH_SIZE,
                                          shuffle=False)
print(f"{100*'='}\nTRAIN DATA: {len(train_dataset.data)}")
print(f"TEST DATA: {len(test_dataset.data)}")
print(f"CLASSES: {test_dataset.classes}\n{100*'='}")

# Show few images with labels
if SHOW_ORIGINAL_IMAGES:
    plot_images(train_dataset, 3)

