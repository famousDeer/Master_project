import matplotlib.pyplot as plt
import torch
import torchvision
import numpy as np
import random

from torch import nn

def set_seed(seed: int=100):
    """Sets random seed for torch operations.

    Args:
        seed (int, optional): Random seed to set. Defaults to 100.
    """
    # Set the seed for general torch operations
    torch.manual_seed(seed)
    # Set the seed for CUDA torch operations 
    torch.cuda.manual_seed(seed)

def plot_images(dataset: torchvision.datasets, num_images: int=3):
    """Plot original images from dataset with labels

    Args:
        dataset (torchvision.datasets): Dataset with images and labels
        num_images (int, optional): Number of how many images to display. 
        It's number of rows and cols. Defaults to 3.
    """
    classes = dataset.classes
    fig, ax = plt.subplots(ncols=num_images, nrows=num_images)
    for row in range(num_images):
        for col in range(num_images):
            i = random.randint(0,len(dataset.data))
            ax[row, col].imshow(dataset[i][0].permute(1,2,0))
            ax[row, col].set_title(classes[dataset[i][1]])
    plt.show()