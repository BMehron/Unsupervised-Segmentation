import random
from matplotlib import pyplot as plt
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import cdist
import importlib
import rhm_map
importlib.reload(rhm_map)
from rhm_map import apply_sinkhorn


from functools import partial
from pathlib import Path
from typing import Optional, Tuple

import cv2
import fire
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from PIL import Image
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from torchvision.utils import draw_bounding_boxes
from tqdm import tqdm
import os
import math 

import extract_utils

def extract_bbox_features_masked_attn(
    images_root: str,
    bbox_file: str,
    model_name: str,
    output_file: str,
):
    """
    Example:
        python extract_bbox_features_masked_attn.py extract_bbox_features_masked_attn \
            --model_name dino_vits16 \
            --images_root "./data/VOC2012/images" \
            --bbox_file "./data/VOC2012/multi_region_bboxes/fixed/bboxes_e2_d5.pth" \
            --output_file "./data/VOC2012/multi_region_bboxes/fixed/bbox_features_e2_d5.pth" \
    """

    # Load bounding boxes
    bbox_list = torch.load(bbox_file)
    total_num_boxes = sum(len(d['bboxes']) for d in bbox_list)
    print(f'Loaded bounding box list. There are {total_num_boxes} total bounding boxes.')

    # Models
    model_name_lower = model_name.lower()
    model, val_transform, patch_size, num_heads = extract_utils.get_model(model_name_lower)
    model.eval().to('cuda')

    # Add hook
    if 'dino' in model_name or 'mocov3' in model_name:
        feat_out = {}
        def hook_fn_forward_qkv(module, input, output):
            feat_out["qkv"] = output
        def hook_fn_forward_last_block_input(module, input, output):
            feat_out["input"] = input
        model._modules["blocks"][-1]._modules["attn"]._modules["qkv"].register_forward_hook(hook_fn_forward_qkv)
        model._modules["blocks"][-1].register_forward_hook(hook_fn_forward_last_block_input)
    else:
        raise ValueError(model_name)
        
    proj = model._modules["blocks"][-1]._modules["attn"]._modules['proj']
    norm2 = model._modules["blocks"][-1]._modules["norm2"]
    mlp = model._modules["blocks"][-1]._modules["mlp"]
    norm = model._modules["norm"]
    def forward_end(x, y):
        y = proj(y)
        x = x + y
        x = x + mlp(norm2(x))
        x = norm(x)
        return x
        
    # Loop over boxes
    for bbox_dict in tqdm(bbox_list):
        # Get image info
        image_id = bbox_dict['id']
        bboxes = bbox_dict['bboxes_original_resolution']
        croped_masks = bbox_dict['croped_masks']
        # Load image as tensor
        image_filename = str(Path(images_root) / f'{image_id}.jpg')
        image = val_transform(Image.open(image_filename).convert('RGB'))  # (3, H, W)
        image = image.unsqueeze(0).to('cuda')  # (1, 3, H, W)
        features_crops = []
        for ((xmin, ymin, xmax, ymax), binary_mask) in zip(bboxes, croped_masks):
            image_crop = image[:, :, ymin:ymax, xmin:xmax]
            flatten_mask = binary_mask.reshape(-1)
            # Image shape
            P = patch_size
            B, C, H, W = image_crop.shape
            H_patch, W_patch = H // P, W // P
            H_pad, W_pad = H_patch * P, W_patch * P
            T = H_patch * W_patch + 1  # number of tokens, add 1 for [CLS]
            # Get q,v, k
            model.get_intermediate_layers(image_crop)[0].squeeze(0)
            input_cls_last_block = feat_out["input"][0][0, 0, :]
            output_qkv = feat_out["qkv"].reshape(B, T, 3, num_heads, -1 // num_heads).permute(2, 0, 3, 1, 4)
            q_cls = output_qkv[0].transpose(1, 2).reshape(B, T, -1)[0, 0, :]
            k = output_qkv[1].transpose(1, 2).reshape(B, T, -1)[0]
            v = output_qkv[2].transpose(1, 2).reshape(B, T, -1)[0]
            attn = q_cls @ k.permute(1, 0) # 1 x 1+N_patches
            attn = attn / math.sqrt(q_cls.shape[0])    
            masked_attn = torch.cat((attn[:1], attn[1:][flatten_mask]), dim=0)
            masked_attn = torch.softmax(masked_attn, dim=-1) 
            masked_cls = masked_attn[0]*v[0] + masked_attn[1:] @ v[1:][flatten_mask]
            masked_cls = forward_end(input_cls_last_block, masked_cls)
            features_crops.append(masked_cls.detach().cpu())
            # attn_weights = torch.softmax(attn, dim=-1)
            # q_cls_updates = attn_weights@v
            # features_crops.append(q_cls_updates.detach().cpu())
        bbox_dict['features'] = torch.stack(features_crops, dim=0)
    
    # Save
    torch.save(bbox_list, output_file)
    print(f'Saved features to {output_file}')
    
    
if __name__ == '__main__':
    fire.Fire(dict(
        extract_bbox_features_masked_attn=extract_bbox_features_masked_attn
    ))
    
    
    