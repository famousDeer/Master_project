import torch
import torchvision

from torch import nn
from torchsummary import summary


class PatchEmbedding(nn.Module):
    def __init__(self,
                 in_channels: int=3, 
                 patch_size: int=16, 
                 embedding_dim: int=768) -> None:
        super(PatchEmbedding, self).__init__()

        self.in_channels = in_channels
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim

        self.patcher = nn.Conv2d(in_channels=self.in_channels,
                                out_channels=self.embedding_dim,
                                kernel_size=self.patch_size,
                                stride=patch_size,
                                padding=0)
        
        self.flatten = nn.Flatten(start_dim=2,
                                  end_dim=3)
        
    def forward(self, x):
        # Check if inputs are the correct shape
        image_resolution = x.shape[-1]
        assert image_resolution % self.patch_size == 0, f"Input image size must be divisible by patch size, image shape: {image_resolution}, patch size: {self.patch_size}"

        x_patched = self.patcher(x)
        x_flattened = self.flatten(x_patched)

        return x_flattened.permute(0,2,1)