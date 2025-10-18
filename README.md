# ImplantBQA

## Usage
### Training
1. Use [ImplantGenerator](https://github.com/wenbc21/ImplantGenerator) to train an implant generation network.
2. Use [get_sagittal_plane.py](get_sagittal_plane.py) to make sagittal dataset from DICOM and STL.
3. Use [labelme](https://github.com/wkentaro/labelme) to label bone in sagittal plane, then use [labelme_to_seg.py](labelme_to_seg.py) to generate 2D segmentation data.
4. Train a 2D segmentation network (e.g. [nnU-Net](https://github.com/MIC-DKFZ/nnUNet)) to segment bone in sagittal planes.

### Inference
1. Use trained [ImplantGenerator](https://github.com/wenbc21/ImplantGenerator) to get implant cylinder.
2. Use [get_sagittal_plane_infer.py](get_sagittal_plane_infer.py) to get sagittal images from DICOM and predicted implants.
3. Use trained 2D segmentation network (e.g. [nnU-Net](https://github.com/MIC-DKFZ/nnUNet)) to get bone segmentation in sagittal planes.
4. Use [quantify.py](quantify.py) to get quantification results.
