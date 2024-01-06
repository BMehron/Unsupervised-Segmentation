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
    

def kmean_bbox_clusters(
    bbox_features_file: str,
    output_file: str,
    num_clusters: int = 21, 
    sim_type: str = 'ot',
    max_iter: int = 20,
    normalize: bool = True,
    ot_temp: int=0.02,
    ot_niter: int=10
):
    """
    Example:
        python optimal_transport_clustering.py kmean_bbox_clusters \
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
    print(f'Computing K-Means clustering with {num_clusters} clusters, sim_type = {sim_type}, max_iter={max_iter}, normalize={normalize}')
    kmeans_ot = KMeansOT(num_clusters = num_clusters,
                     sim_type=sim_type, 
                     max_iters=max_iter, 
                     ot_temp=ot_temp,
                     ot_niter = ot_niter,
                     normalize=normalize)
    clusters = kmeans_ot.fit_predict(torch.from_numpy(all_features))
    
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
        kmean_bbox_clusters=kmean_bbox_clusters
    ))