# ImplantTerheyden

## Training
(datasets/sagittal_image_label)
1. run get_sagittal_plane.py, use GT DICOM and STL to get sagittal plane
2. use labelme to label bone in sagittal plane
3. run labelme_to_seg.py, use labeled mask to generate segment data
4. train nnUNet to segment bone
5. run seg_vis.py to transform segment results
```
export nnUNet_raw="/home/amax/Project/wbc/implant_and_classify/nnUNet/dataset/nnUNet_raw"
export nnUNet_preprocessed="/home/amax/Project/wbc/implant_and_classify/nnUNet/dataset/nnUNet_preprocessed"
export nnUNet_results="/home/amax/Project/wbc/implant_and_classify/nnUNet/dataset/nnUNet_results"

nnUNetv2_plan_and_preprocess -d 812 --verify_dataset_integrity
nnUNetv2_train 812 2d 0
nnUNetv2_predict -i dataset/imagesTs -o dataset/predictTs -d 812 -c 2d -f 0
```

(ImplantDifffusion)
1. get data
2. train ImplantDiffusion
3. get predict cylinder

(ImplantTerheyden)
1. run Terheyden.py to get results

## Inference
1. run ImplantDiffusion to get cylinder mask from DICOM
2. run get_sagittal_plane.py to get sagittal plane from DICOM and cylinder
3. run nnUNet to get bone segmentation
4. run Terheyden.py to get classification from cylinder and bone