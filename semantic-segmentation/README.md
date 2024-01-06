## Semantic Segmentation

We begin by extracting features and eigenvectors from our images. For instructions on this process, follow the steps in "Extraction" in the main `README`. 

Next, we obtain coarse (i.e. patch-level) semantic segmentations following **Deep Spectral Method* paper approach. This process involves (1) extracting segments from the eigenvectors, (2) taking a bounding box around them, (3) extracting features for these boxes, (4) clustering these features, (5) obtaining coarse semantic segmentations. 
```

For example, you can run the following in the `extract` directory. 

```bash
# Example parameters for the semantic segmentation experiments
DATASET="VOC2012"
MODEL="dino_vits16"
MATRIX="laplacian"
DOWNSAMPLE=16
N_SEG=15
N_ERODE=2
N_DILATE=5

# 1. Extract segments
python extract.py extract_multi_region_segmentations \
    --non_adaptive_num_segments ${N_SEG} \
    --features_dir "./data/${DATASET}/features/${MODEL}" \
    --eigs_dir "./data/${DATASET}/eigs/${MATRIX}" \
    --output_dir "./data/${DATASET}/multi_region_segmentation/${MATRIX}"

# 2. Extract bounding boxes
python extract.py extract_bboxes \
    --features_dir "./data/${DATASET}/features/${MODEL}" \
    --segmentations_dir "./data/${DATASET}/multi_region_segmentation/${MATRIX}" \
    --num_erode ${N_ERODE} \
    --num_dilate ${N_DILATE} \
    --downsample_factor ${DOWNSAMPLE} \
    --output_file "./data/${DATASET}/multi_region_bboxes/${MATRIX}/bboxes.pth"

# 3. Extract bounding box features
python extract.py extract_bbox_features \
    --model_name ${MODEL} \
    --images_root "./data/${DATASET}/images" \
    --bbox_file "./data/${DATASET}/multi_region_bboxes/${MATRIX}/bboxes.pth" \
    --output_file "./data/${DATASET}/multi_region_bboxes/${MATRIX}/bbox_features.pth"

# 4. Extract clusters
python extract.py extract_bbox_clusters \
    --bbox_features_file "./data/${DATASET}/multi_region_bboxes/${MATRIX}/bbox_features.pth" \
    --output_file "./data/${DATASET}/multi_region_bboxes/${MATRIX}/bbox_clusters.pth" 

# 5. Create semantic segmentations
python extract.py extract_semantic_segmentations \
    --segmentations_dir "./data/${DATASET}/multi_region_segmentation/${MATRIX}" \
    --bbox_clusters_file "./data/${DATASET}/multi_region_bboxes/${MATRIX}/bbox_clusters.pth" \
    --output_dir "./data/${DATASET}/semantic_segmentations/patches/${MATRIX}/segmaps" 
```

At this point, you can evaluate the segmentations using `eval.py` in this directory. For example:
```bash
python eval.py segments_dir="/output_dir/from/above"
```

In step (1), if you want to extract multi region segmentations using **Adaptive Concept Generator (ACG)**, run this command instead of command 1:
```bash
python ACG.py extract_multi_region_segmentations \
    --features_dir "./data/${DATASET}/features/${MODEL}/all_features.pth" \
    --output_dir "./data/${DATASET}/multi_region_segmentation/${MATRIX}"
```

In step (3), if you want to extract the **masked attention** for the DINO CLS instead of just the CLS token, run this command instead of command 3:
```bash
python dsm_bbox_masked_attn.py extract_bbox_features_masked_attn \
            --model_name ${MODEL} \
            --images_root "./data/${DATASET}/images" \
            --bbox_file "./data/${DATASET}/multi_region_bboxes/${MATRIX}/bboxes.pth" \
            --output_file "./data/${DATASET}/multi_region_bboxes/${MATRIX}/bbox_features.pth"
```


In step (4), if you want to run **Optimal Transport Clustering**, run this command instead of command 4:
```bash
python custom_clusterings.py bbox_clusters \
            --bbox_features_file "./data/${DATASET}/multi_region_bboxes/${MATRIX}/bbox_features.pth" \
            --output_file "./data/${DATASET}/multi_region_bboxes/${MATRIX}/bbox_clusters.pth"
            --method_of_clustering "otc"

```

In step (4), if you want to run **Attention-Based-Clustering**, run this command instead of command 4:
```bash
python custom_clusterings.py bbox_clusters \
            --bbox_features_file "./data/${DATASET}/multi_region_bboxes/${MATRIX}/bbox_features.pth" \
            --output_file "./data/${DATASET}/multi_region_bboxes/${MATRIX}/bbox_clusters.pth"
            --method_of_clustering "abc"

Please note that the unsupervised semantic segmentation results have very high variance; some runs are much better than others. This variance is primarily due to the random seeds of the K-means clustering steps above, and it is secondarily due to randomness in the self-training stage. Also please note that this code has been heavily re-factored for its public release. Although we try to ensure that there are no bugs, it is nevertheless possible that there is a bug we have overlooked. 
