import torch
import torchvision

from torch import nn
from torchsummary import summary


class PatchEmbedding(nn.Module):
    """Turns a 2D input image into a 1D sequence learnable embedding vector.
    Using simple Convolution and Flatten from nn.
    
    Args:
        in_channels (int): Number of color channels for the input images. Defaults to 3.
        patch_size (int): Size of patches to convert input image into. Defaults to 16.
        embedding_dim (int): Size of embedding to turn image into. Defaults to 768.
    """
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

class MultiheadSelfAttentionBlock(nn.Module):
    def __init__(self,
                 embedding_dim: int=768,
                 num_heads: int=12,
                 dropout: float=0.0) -> None:
        super(MultiheadSelfAttentionBlock, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.dropout = dropout
        
        self.layer_norm = nn.LayerNorm(normalized_shape=self.embedding_dim)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=self.embedding_dim,
                                                    num_heads=self.num_heads,
                                                    dropout=self.dropout,
                                                    batch_first=True)
    
    def forward(self, x):
        x = self.layer_norm(x)
        attn_output, _ = self.multihead_attn(query=x,
                                             key=x,
                                             value=x,
                                             need_weights=False)
        
        return attn_output

class MLPBlock(nn.Module):
    def __init__(self,
                 embedding_dim: int=768,
                 mlp_size: int=3072,
                 dropout: float=0.1) -> None:
        super(MLPBlock, self).__init__()

        self.embedding_dim = embedding_dim
        self.mlp_size = mlp_size
        self.dropout = dropout

        self.layer_norm = nn.LayerNorm(normalized_shape=self.embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=self.embedding_dim,
                      out_features=self.mlp_size),
            nn.GELU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(in_features=self.mlp_size,
                      out_features=self.embedding_dim),
            nn.Dropout(p=self.dropout)
        )

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.mlp(x)
        
        return x
    
class ViT(nn.Module):
    def __init__(self,
                 img_height: int,
                 img_width: int,
                 batch_size: int,
                 patch_size: int=16,
                 embedding_dim: int=768) -> None:
        super(ViT,self).__init__()
        
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width
        self.patch_size = patch_size
        self.number_patches = (self.img_height * self.img_width) / self.patch_size ** 2
        self.embedding_dim = embedding_dim
        
        self.class_embedding = nn.Parameter(data=torch.randn(self.batch_size, 
                                                             1, 
                                                             self.embedding_dim),
                                            requires_grad=True)
        self.position_embedding = nn.Parameter(data=torch.randn(1,
                                                                self.number_patches+1,
                                                                self.embedding_dim),
                                               requires_grad=True)
        
    def forward(self, x):
        
        return x