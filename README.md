<div align="center">
 
## Semester Project: Unsupervised Segmentation


</div>

### Description
This code accompanies the final report. 

### How to run   

#### Dependencies
The minimal set of dependencies is listed in `requirements.txt`.

#### Data Preparation

The data preparation process simply consists of collecting your images into a single folder. Here, we describe the process for [Pascal VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012//). Pascal VOC 2007 and MS-COCO are similar. 

Download the images into a single folder. Then create a text file where each line contains the name of an image file. For example, here is our initial data layout:
```
data
└── VOC2012
    ├── images
    │   └── {image_id}.jpg
    └── lists
        └── images.txt
```

#### Extraction

We first extract features from images and stores these into files. We then extract eigenvectors from these features. Once we have the eigenvectors, we can perform downstream tasks such as object segmentation and object localization. 

The primary script for this extraction process is `extract.py` in the `extract/` directory. All functions in `extract.py` have helpful docstrings with example usage. 

##### Step 1: Feature Extraction

First, we extract features from our images and save them to `.pth` files. 

With regard to models, our repository currently only supports DINO, but other models are easy to add (see the `get_model` function in `extract_utils.py`). The DINO model is downloaded automatically using `torch.hub`. 

Here is an example using `dino_vits16`:

```bash
python extract.py extract_features \
    --images_list "./data/VOC2012/lists/images.txt" \
    --images_root "./data/VOC2012/images" \
    --output_dir "./data/VOC2012/features/dino_vits16" \
    --model_name dino_vits16 \
    --batch_size 1
```

##### Step 2: Eigenvector Computation

Second, we extract eigenvectors from our features and save them to `.pth` files. 

Here, we extract the top `K=5` eigenvectors of the Laplacian matrix of our features:

```bash
python extract.py extract_eigs \
    --images_root "./data/VOC2012/images" \
    --features_dir "./data/VOC2012/features/dino_vits16" \
    --which_matrix "laplacian" \
    --output_dir "./data/VOC2012/eigs/laplacian" \
    --K 5
```

The final data structure after extracting eigenvectors looks like:
```
data
├── VOC2012
│   ├── eigs
│   │   └── {outpur_dir_name}
│   │       └── {image_id}.pth
│   ├── features
│   │   └── {model_name}
│   │       └── {image_id}.pth
│   ├── images
│   │   └── {image_id}.jpg
│   └── lists
│       └── images.txt
└── VOC2007
    └── ...
```

At this point, you are ready to use the eigenvectors for downstream task semantic segmentation. 


#### Semantic Segmentation

For semantic segmentation, we provide full instructions in the `semantic-segmentation` subfolder.
