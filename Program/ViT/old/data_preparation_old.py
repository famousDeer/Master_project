import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import json
import random
import cv2
import numpy as np

from tqdm import tqdm
from pathlib import Path
from data_setup import CustomDataSet
from collections import defaultdict
from Program.SSD.models import TinyVGG
try:
    import torch
    import torchvision
    assert int(torch.__version__.split('.')[1]) >= 12, "torch version should be 1.12+"
    assert int(torchvision.__version__.split('.')[1]) >= 13, "torchvision version should be 0.13+"
    print(f"torch version {torch.__version__}")
    print(f"torchvision version {torchvision.__version__}")
except:
    print(f"[INFO] torch/torchvision versions not as required, installinf nightly version...")
    os.system("pip3 install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117")
    import torch
    import torchvision
    print(f"torch version {torch.__version__}")
    print(f"torchvision version {torchvision.__version__}")

try:
    from torchinfo import summary
except:
    print(f"[INFO] Couldn't find torchinfo...\nInstalling it...")
    os.system("pip3 install -q torchinfo")
    from torchinfo import summary

from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader

def Category_reader(path: str):
    with open(path, 'r') as f:
        text = json.load(f)
    categories = text['categories']
    class_names = {}
    for category in categories:
        class_names[category['id']] = [category['supercategory'], category['name']]
    return class_names

def resize_bbox(path: str, image_size: int, coco):
    """Function resizing boundary box coordinates according to original and new image size.

    Args:
        path (str): Path to json file with annotations
        image_size (int): New Image size
        coco (_type_): COCO dataset
    """
    with open(path, 'r') as f:
        text = json.load(f)
    images = text['images']
    img_org_size = {}
    for image in tqdm(images):
        id = image['id']
        height = image['height']
        width = image['width']
        img_org_size[id] = [height, width]
    for anns in tqdm(range(len(coco))):
        for ann in range(len(coco[anns][1])):
            height, width = img_org_size[coco[anns][1][ann]['image_id']]
            bbox = coco[anns][1][ann]['bbox']
            bbox[0] *= image_size/width
            bbox[1] *= image_size/height
            bbox[2] *= image_size/width
            bbox[3] *= image_size/height
            coco[anns][1][ann]['bbox'] = bbox

# Setup device-agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

# Paths for training and validation images and annotations 
train_image_path = Path("/home/famousdeer/Desktop/Praca magisterska/Program/data/Images/train2017")
train_ann_path = Path("/home/famousdeer/Desktop/Praca magisterska/Program/data/annotations/instances_train2017.json")
val_image_path = Path("/home/famousdeer/Desktop/Praca magisterska/Program/data/Images/val2017")
val_ann_path = Path("/home/famousdeer/Desktop/Praca magisterska/Program/data/annotations/instances_val2017.json")

# Get class names 
class_names = Category_reader(val_ann_path)

# Creating image size base on tabel 3 in the ViT paper
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4

# Create transform pipeline manually
manual_transform = transforms.Compose([
                                        transforms.Resize((IMG_SIZE, IMG_SIZE)),
                                        transforms.ToTensor()
                                        ])
print(f"Manually created transforms: {manual_transform}")

# Load COCO dataset
train_coco = CustomDataSet(root=train_image_path, annFile=train_ann_path, transform=manual_transform)
val_coco = CustomDataSet(root=val_image_path, annFile=val_ann_path, transform=manual_transform)

# Resize boundary box according to resizing image
resize_bbox(val_ann_path, IMG_SIZE, val_coco)
resize_bbox(train_ann_path, IMG_SIZE, train_coco)

# Use DataLoader to dataset with batch size
train_loader = DataLoader(
                        dataset=train_coco,
                        batch_size=BATCH_SIZE,
                        shuffle=True,
                        num_workers=NUM_WORKERS,
                        pin_memory=True,
                        collate_fn=lambda x: x
                        )
val_loader = DataLoader(
                        dataset=val_coco,
                        batch_size=BATCH_SIZE,
                        shuffle=False,
                        num_workers=NUM_WORKERS,
                        pin_memory=True,
                        collate_fn=lambda x: x
                        )

# print(train_loader.dataset, val_loader.dataset)

# VISUALIZATION IMAGE FROM DATALOADER

iter_image_batch = iter(train_loader.dataset)
image_batch = next(iter_image_batch)

image, labels = image_batch[0], image_batch[1]



title = ''
img = image.numpy().transpose(1,2,0)
img = img.copy()
plt.imshow(img)
ax = plt.gca()
for label in labels:
    title += class_names[label['category_id']][1] + ' '
    bbox = label['bbox']
    print(label)
    rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

ax.add_patch(rect)
# plt.imshow(image.permute(1,2,0))
plt.title(title)
plt.show()