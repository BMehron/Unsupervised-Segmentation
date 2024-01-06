from functools import partial
from pathlib import Path
from typing import Optional, Tuple

import cv2
import fire
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from accelerate import Accelerator
from PIL import Image
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from torchvision.utils import draw_bounding_boxes
from tqdm import tqdm
#from rhm_map import apply_sinkhorn
import random

import extract_utils

import torch
from torch import Tensor, nn

import math
from typing import Tuple, Type
    
class Attention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        # Get output
        out = attn @ v # B x N_heads x N_tokens x C_per_head
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out

class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.ReLU,
        skip_connection: bool = False,
        layer_norm: bool = False,
        dropout: float = 0.1
        
    ) -> None:
        super().__init__()
        
        self.skip_connection = skip_connection
        self.layer_norm = layer_norm
        
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = act()
        
        if self.layer_norm:
            self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mlp_out = self.lin2(self.dropout(self.act(self.lin1(x))))
        if self.skip_connection:
            mlp_out = x + mlp_out
        if self.layer_norm:
            mlp_out = self.norm(mlp_out)
        return mlp_out
    
    
class KMeanAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        num_clusters: int = 21,
        activation: Type[nn.Module] = nn.ReLU,
        skip_connection: bool = True,
        layer_norm: bool = True,
        dropout: float = 0.1
    ) -> None:
        """
        Arguments:
          embedding_dim (int): the channel dimension of the embeddings
          num_heads (int): the number of heads in the attention layers
          mlp_dim (int): the hidden dimension of the mlp block
          num_clusters (int): number of clusters
          activation (nn.Module): the activation of the mlp block
        """
        super().__init__() 
        # Initialize claster centers embedings
        self.cluster_centers = torch.nn.Parameter(data=torch.randn(num_clusters, embedding_dim) / math.sqrt(embedding_dim / 2), requires_grad=True)
        
        # Attention Block
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation, skip_connection, layer_norm, dropout=dropout)
        self.norm2 = nn.LayerNorm(embedding_dim)

    def forward(
        self, bbox_features: Tensor) -> Tuple[Tensor, Tensor]:
        # Self attention block
        attn_out = self.self_attn(q=bbox_features, k=self.cluster_centers.unsqueeze(dim=0), v=self.cluster_centers.unsqueeze(dim=0))
        updated_clusters = bbox_features + attn_out
        updated_clusters = self.norm1(updated_clusters)

        # MLP block
        updated_clusters = self.mlp(updated_clusters)
        
        return updated_clusters.squeeze(), self.cluster_centers
    
    
    
class ACGBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 256,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 1,
        skip_connection: bool = True,
        layer_norm: bool = True,
        dropout: float = 0.1,
        use_self_attn: bool = True
    ) -> None:
        """
        A ACG block with 3 layers: (1) cross-attention of 
        q=image_embedding, k=v=prototypes, (2) self-attention of prototypes, (3) mlp
        block on prototypes
        
        Arguments:
          embedding_dim (int): the channel dimension of the embeddings
          num_heads (int): the number of heads in the attention layers
          mlp_dim (int): the hidden dimension of the mlp block
          activation (nn.Module): the activation of the mlp block
        """
        super().__init__() 
        
        self.use_self_attn = use_self_attn
        
        # Cross Attention Block
        self.cross_attn_concepts_to_image = Attention(embedding_dim, num_heads)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embedding_dim)
        
        # Self Attention Block
        if self.use_self_attn:
            self.self_attn = Attention(embedding_dim, num_heads)
            self.dropout2 = nn.Dropout(dropout)
            self.norm2 = nn.LayerNorm(embedding_dim)
        
        #MLP
        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation, skip_connection, layer_norm, dropout)
        
    def forward(self, prototypes, image_embedding) -> Tensor:
        # Cross attention block
        cross_attn = self.dropout1(self.cross_attn_concepts_to_image(q=prototypes, k=image_embedding, v=image_embedding))
        updated_prototypes = self.norm1(prototypes + cross_attn)
        
        # Self attention blockv
        if self.use_self_attn:
            self_attn = self.dropout2(self.self_attn(q=updated_prototypes, k=updated_prototypes, v=updated_prototypes))
            updated_prototypes = self.norm2(updated_prototypes + self_attn)
        
        updated_prototypes = self.mlp(updated_prototypes) # ADD RES
        

class ACGTransformer(nn.Module):
    def __init__(
        self,
        n_prototypes: Tensor,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 1,
        use_self_attn: bool = True
    ) -> None:
        """
        A ACG transformer that attends to an input image pixel-representation using prototypes as
        queries .

        Args:
          initial_prototypes (torch.Tensor): initial prototypes.
          depth (int): number of layers in the transformer
          embedding_dim (int): the channel dimension for the input embeddings
          num_heads (int): the number of heads for multihead attention. Must
            divide embedding_dim
          mlp_dim (int): the channel dimension internal to the MLP block
          activation (nn.Module): the activation to use in the MLP block
        """
        super().__init__()

        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.use_self_attn = use_self_attn
        
        self.initial_prototypes = torch.nn.Parameter(data=torch.randn(n_prototypes, embedding_dim) / math.sqrt(embedding_dim / 2), requires_grad=True)
        self.layers = nn.ModuleList()

        for i in range(depth):
            self.layers.append(
                ACGBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    use_self_attn=use_self_attn
                )
            )

    def forward(
        self,
        image_embedding: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
          image_embedding (torch.Tensor): image to attend to. Should be shape
            B x hw x embedding_dim for any h and w.

        Returns:
          torch.Tensor: adapted prototypes 
        """
        # BxCxHxW -> BxCxHW == B x N_image_pixels x C
        # bs, c, h, w = image_embedding.shape
        # image_embedding = image_embedding.flatten(2).permute(0, 2, 1)

        # Prepare prototypes. Copy prototypes batch times: K x C -> B x K x C
        prototypes = self.initial_prototypes.unsqueeze(0).repeat(image_embedding.shape[0], 1, 1)
        prototypes = F.normalize(prototypes, dim=-1)

        # Apply ACG transformer blocks
        for layer in self.layers:
            prototypes = layer(
                prototypes=prototypes,
                image_embedding=image_embedding
            )
            
        return prototypes