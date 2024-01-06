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

from tqdm import tqdm
import os

from transformer_archs import KMeanAttentionBlock
from torchmetrics.functional.pairwise import pairwise_cosine_similarity
from torch.linalg import matrix_norm

def attention_based_clustering(data, model_name='ABC_k_2_k_05_lr_3e5_500_epochs_21_clusters'):
    model = KMeanAttentionBlock(num_clusters = 21, embedding_dim=384, num_heads=1, mlp_dim=512, layer_norm=False, skip_connection=False)
    model.load_state_dict(torch.load('{}.pth'.format(model_name)))
    model.to('cuda')
    model.eval()
    
    BATCH_SIZE = 4096
    
    clusters = []
    output = None

    for i, batch_id in enumerate(tqdm(range(0, len(data), BATCH_SIZE), position=0, leave=True)):
        batch = data[batch_id:batch_id+BATCH_SIZE].unsqueeze(dim=0).to('cuda')
        features, cluster_centers = model(batch)
        output = F.relu(pairwise_cosine_similarity(features, cluster_centers))
        clusters.extend(list(map(lambda x: x.item(), list(torch.argmax(output, dim=-1).cpu()))))
    return clusters


class KMeansOT:
    def __init__(self, num_clusters, sim_type='l2', max_iters=10, normalize=False ,ot_temp=0.02, ot_niter=40):
        self.num_clusters = num_clusters
        self.sim_type = sim_type
        self.max_iters = max_iters
        self.normalize = normalize
        self.ot_temp = ot_temp
        self.ot_niter = ot_niter
        self.cluster_centers = None

    def fit_predict(self, X):
        if self.normalize:
            X = (X - X.mean(dim=0, keepdims=True)) / X.std(dim=0, keepdims=True)
        self.cluster_centers = X[random.sample(range(len(X)), self.num_clusters)].detach().clone()
        X_norm = F.normalize(X, dim=1)

        for _ in tqdm(range(self.max_iters)):
            if self.sim_type == 'l2':
                new_labels = ((X[:, None, :] - self.cluster_centers)**2).sum(axis=-1).argmin(dim=1)
                self.cluster_centers = torch.stack([X[new_labels == k].mean(dim=0) for k in range(self.num_clusters)])
            elif self.sim_type in ['cos', 'ot']:
                cluster_centers_norm = F.normalize(self.cluster_centers, dim=1)
                similarity = cluster_centers_norm @ X_norm.T  # K x N

                if self.sim_type == 'ot':
                    similarity = apply_sinkhorn(similarity.float().cuda(), eps=self.ot_temp, niter=self.ot_niter)[0].detach().cpu() # K x N
                    self.cluster_centers = similarity.float() @ X
                else:
                    new_labels = similarity.argmax(dim=0)
                    self.cluster_centers = torch.stack([X[new_labels == k].mean(dim=0) for k in range(self.num_clusters)])
            else:
                raise Exception('Choose correct similarity function')
                
        labels = None
        if self.sim_type == 'l2':
            labels = ((X[:, None, :] - self.cluster_centers)**2).sum(axis=-1).argmin(dim=1)
        else:
            cluster_centers_norm = F.normalize(self.cluster_centers, dim=1)
            similarity = cluster_centers_norm @ X_norm.T  # K x N
            labels = similarity.argmax(dim=0)
        return labels
    

def bbox_clusters(
    bbox_features_file: str,
    output_file: str,
    method_of_clustering: str='otc',
    num_clusters: int = 21, 
    sim_type: str = 'ot',
    max_iter: int = 20,
    normalize: bool = True,
    ot_temp: int=0.02,
    ot_niter: int=10
):
    """
    Example:
        python custom_clusterings.py bbox_clusters \
            --bbox_features_file "./data/VOC2012/multi_region_bboxes/fixed/bbox_features_e2_d5.pth" \
            --output_file "./data/VOC2012/multi_region_bboxes/fixed/bbox_clusters_e2_d5_pca_32.pth" \
    """
    # Load bounding boxes
    bbox_list = torch.load(bbox_features_file)
    total_num_boxes = sum(len(d['bboxes']) for d in bbox_list)
    print(f'Loaded bounding box list. There are {total_num_boxes} total bounding boxes with features.')

    # Loop over boxes and stack features with PyTorch, because Numpy is too slow
    print(f'Stacking and normalizing features')
    all_features = torch.cat([bbox_dict['features'] for bbox_dict in bbox_list], dim=0)  # (numBbox, D)
    all_features = all_features / torch.norm(all_features, dim=-1, keepdim=True)  # (numBbox, D)f
    all_features = all_features.numpy()

    # Cluster: K-Means
    print(f'Computing K-Means clustering with {num_clusters} clusters, method_of_clustering = {method_of_clustering}')
    if method_of_clustering == 'otc':
        kmeans_ot = KMeansOT(num_clusters = num_clusters,
                         sim_type=sim_type, 
                         max_iters=max_iter, 
                         ot_temp=ot_temp,
                         ot_niter = ot_niter,
                         normalize=normalize)
        clusters = kmeans_ot.fit_predict(torch.from_numpy(all_features))
    elif method_of_clustering == 'abc':
        clusters = attention_based_clustering(torch.from_numpy(all_features))
    else:
        raise Exception("Sorry, choose correct method for clustering!")
    
    # Print 
    _indices, _counts = np.unique(clusters, return_counts=True)
    print(f'Cluster indices: {_indices.tolist()}')
    print(f'Cluster counts: {_counts.tolist()}')

    # Loop over boxes and add clusters
    idx = 0
    for bbox_dict in bbox_list:
        num_bboxes = len(bbox_dict['bboxes'])
        del bbox_dict['features']  # bbox_dict['features'] = bbox_dict['features'].squeeze()
        bbox_dict['clusters'] = clusters[idx: idx + num_bboxes]
        idx = idx + num_bboxes
    
    # Save
    torch.save(bbox_list, output_file)
    print(f'Saved features to {output_file}')
    
if __name__ == '__main__':
    fire.Fire(dict(
        bbox_clusters=bbox_clusters
    ))