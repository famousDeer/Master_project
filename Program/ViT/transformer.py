import torch
import torchvision
import matplotlib.pyplot as plt
import os
import random

from torch import nn
from torchvision import transforms
from torchinfo import summary
from helper_functions import plot_images, set_seed
from pathlib import Path
from models import PatchEmbedding

# Choose device 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'DEVICE: {device}')

# Set random seed
set_seed(seed=100)

# Custom parameters
SHOW_ORIGINAL_IMAGES = False
HEIGHT = 320
WIDTH = 320
COLOR_CHANNELS = 3
PATCH_SIZE = 16
NUMBER_OF_PATCHES = int((HEIGHT * WIDTH) / PATCH_SIZE**2)
PATCHES_PER_ROW = HEIGHT // PATCH_SIZE
EMBEDDING_DIM = PATCH_SIZE * PATCH_SIZE * COLOR_CHANNELS
print(f"Number of patches (N) with image height (H={HEIGHT}), width (W={WIDTH}) and patch size (P={PATCH_SIZE}): {NUMBER_OF_PATCHES}")

# Hyper-parameters
NUM_EPOCHS = 10
BATCH_SIZE = 1024
LEARNING_RATE = 0.001

# Image preprocessing modules
transform = transforms.Compose([
    transforms.Resize((HEIGHT,WIDTH)),
    transforms.Pad(4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(320),
    transforms.ToTensor()])
print(f"Used transform: {transform}")
ROOT = Path('/home/famousdeer/Desktop/Praca magisterska/Program/data/Cityscape')
# CIFAR-10 dataset
# train_dataset = torchvision.datasets.CIFAR10(root='../data/',
#                                              train=True, 
#                                              transform=transform,
#                                              download=True)

# test_dataset = torchvision.datasets.CIFAR10(root='../data/',
#                                             train=False, 
#                                             transform=transforms.ToTensor())

train_dataset = torchvision.datasets.Cityscapes(root=ROOT,
                                                split='train',
                                                mode='fine',
                                                target_type='instance',
                                                transform=transform)

test_dataset = torchvision.datasets.Cityscapes(root=ROOT,
                                               split='test',
                                               mode='fine',
                                               target_type='instance',
                                               transform=transform)

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=BATCH_SIZE,
                                           shuffle=True,
                                           num_workers=os.cpu_count())

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=BATCH_SIZE,
                                          shuffle=False,
                                          num_workers=os.cpu_count())

print(f"{100*'='}\nTRAIN DATA: {len(train_dataset)}")
print(f"TEST DATA: {len(test_dataset)}")
print(f"CLASSES: {test_dataset.classes}\n{100*'='}")

###SHOW_SOME_RESULTS###

# Show few images with labels
if SHOW_ORIGINAL_IMAGES:
    plot_images(train_dataset, 3)
# Original Image
plt.imshow(train_dataset[0][0].permute(1,2,0))

conv2d = nn.Conv2d(in_channels=3,
                   out_channels=768,
                   kernel_size=PATCH_SIZE,
                   stride=PATCH_SIZE,
                   padding=0)
random_indexes = random.sample(range(0, 768), k=25)
print(f"Showing random convolutional feature maps from indexes: {random_indexes}")

fig, axs = plt.subplots(nrows=5, ncols=5, figsize=(12,12))

# Turn image into feature maps
image_out_of_conv = conv2d(train_dataset[0][0].unsqueeze(0))
print(f"Image feature map shape: {image_out_of_conv.shape}")

# Flatten the feature maps on height and width
flatten = nn.Flatten(start_dim=2,
                     end_dim=3)
image_out_of_flatten = flatten(image_out_of_conv)
image_out_of_flatten_reshaped = image_out_of_flatten.permute(0,2,1)
print(f"Flattend image feature map shape: {image_out_of_flatten_reshaped.shape}")

single_flattened_feature_map = image_out_of_flatten_reshaped[:,:,0]
plt.figure(figsize=(20,20))
plt.imshow(single_flattened_feature_map.detach().numpy())
plt.axis('off')
for row in range(5):
    for col in range(5):
        image_conv_feature_map=image_out_of_conv[:,random_indexes[row+col],:,:]
        axs[row,col].imshow(image_conv_feature_map.squeeze().detach().numpy())
        axs[row,col].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        axs[row,col].set_title(random_indexes[5*row+col])
plt.show()

####START_MODEL####
# Create an instance of patch embedding layer
patch_embed = PatchEmbedding(in_channels=COLOR_CHANNELS,
                             patch_size=PATCH_SIZE,
                             embedding_dim=EMBEDDING_DIM)

# Check if everything is correct 
print(f"Input image shape: {train_dataset[0][0].unsqueeze(0).shape}")
patch_embedded_image = patch_embed(train_dataset[0][0].unsqueeze(0))
print(f"Output patch embedding shape: {patch_embedded_image.shape}")

# Model summary for single image 
summary(model=PatchEmbedding(),
        input_size=train_dataset[0][0].unsqueeze(0).shape,
        col_names=['input_size', 'output_size', 'num_params', 'trainable'],
        col_width=20,
        row_settings=['var_names'])

