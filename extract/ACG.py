from functools import partial
from pathlib import Path
from typing import Optional, Tuple

import cv2
import fire
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from tqdm import tqdm

from transformer_archs import ACGTransformer

torch.multiprocessing.set_sharing_strategy('file_system')

import extract_utils

import gc

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
print(device)

def clear_gpu():
    torch.cuda.empty_cache()
    print(gc.collect())
    

    
def extract_multi_region_segmentations(model_path='ACG_3e4_400_6_6_6', 
                                       upsample_befor_hard_assig = False, 
                                       features_dir = 'data/VOC2012/features/dino_vits16/all_features.pth', 
                                       output_dir='./data/VOC2012/multi_region_segmentation/ACSeg', 
                                       infer_bg_index=True):
    
    model = ACGTransformer(n_prototypes = 6,
                       depth = 6,
                       embedding_dim = 384,
                       num_heads = 6,
                       mlp_dim = 256,
                       use_self_attn = True)

    model.load_state_dict(torch.load('{}.pth'.format(model_path)))
    model.to('cuda').eval()
    
    extract_utils.make_output_dir(output_dir)
    all_features = torch.load(features_dir)
    all_features['prototypes'] = torch.tensor([])
    image_features = torch.stack(all_features['k']).squeeze()[:, 1:, :]
    batch_size = 1024
    for i in tqdm(range(0, len(image_features), batch_size)):
        batch = image_features[i:i+batch_size].to(device)
        adjusted_prototypes = model(batch)
        batch_norm = F.normalize(batch, dim=2) # BxN_pixelsxC
        prototypes_norm = F.normalize(adjusted_prototypes, dim=2) # BxKxC
        soft_assig = (batch_norm@prototypes_norm.permute(0, 2, 1)).reshape(len(batch), NUM_PATCHES, NUM_PATCHES, -1) # BxN_PxN_PxK
        # TODO: Improve this step in the pipeline.
        # Background detection: we assume that the segment with the most border pixels is the 
        # background region. We will always make this region equal 0. 
        for j, idx in enumerate(all_features['id'][i:i+batch_size]):
            cur_soft_assig = soft_assig[j].detach().cpu().numpy()
            H_org, W_org = all_features['old_shape'][i+j]
            if upsample_befor_hard_assig:
                cur_soft_assig = cv2.resize(cur_soft_assig, dsize=(W_org, H_org), interpolation=cv2.INTER_NEAREST)
            cur_segmap = np.argmax(cur_soft_assig, axis=-1)
            output_file = str(Path(output_dir) / f'{idx}.png')
            if infer_bg_index:
                indices, normlized_counts = extract_utils.get_border_fraction(cur_segmap)
                bg_index = indices[np.argmax(normlized_counts)].item()
                bg_region = (cur_segmap == bg_index)
                zero_region = (cur_segmap == 0)
                cur_segmap[bg_region] = 0
                cur_segmap[zero_region] = bg_index   
                # permute prototypes also 
                adjusted_prototypes[j, [bg_index, 0]] = adjusted_prototypes[j, [0, bg_index]]            
            
            Image.fromarray(cur_segmap.astype(np.uint8)).convert('L').save(output_file)
        
        all_features['prototypes'] = torch.cat((all_features['prototypes'], adjusted_prototypes.detach().cpu()))
            
    torch.save(all_features, features_dir)
    
    
if __name__ == '__main__':
    fire.Fire(dict(
        extract_multi_region_segmentations=extract_multi_region_segmentations
    ))