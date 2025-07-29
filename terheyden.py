import json
import os
import PIL.Image
import shutil
import pandas as pd
import random
import cv2
import numpy as np

def terheyden(seg_dir, implant_dir, out_dir):
    
    seg_files = [item.path for item in os.scandir(seg_dir) if item.is_file() and item.path.endswith("png")]
    # implant_files = [item.path for item in os.scandir(implant_dir) if item.is_file() and "implant" in item.path]
    seg_files.sort()
    # implant_files.sort()
    
    os.makedirs(f"{out_dir}/vis_seg", exist_ok=True)
    os.makedirs(f"{out_dir}/vis_imp", exist_ok=True)
    os.makedirs(f"{out_dir}/vis_cmb", exist_ok=True)
    os.makedirs(f"{out_dir}/vis_cmb_mod", exist_ok=True)

    for i in range(len(seg_files)):
        # load images
        seg_file = seg_files[i]
        image_name = os.path.split(seg_file)[-1].split("_")[0]
        implant_file = os.path.join(implant_dir, f"{image_name}_implant.png")
        image_name = os.path.split(seg_file)[-1].split("_")[0]
        seg_img = cv2.imread(seg_file, cv2.IMREAD_GRAYSCALE)
        imp_img = cv2.imread(implant_file, cv2.IMREAD_GRAYSCALE)
        imp_img[imp_img == 255] = 1
        # imp_img = cv2.rotate(imp_img, cv2.ROTATE_180)
        print(image_name, seg_file, implant_file)
        
        # segment
        seg_contours, _ = cv2.findContours(seg_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        seg_img_vis = cv2.cvtColor(seg_img, cv2.COLOR_GRAY2BGR) * 255
        cv2.drawContours(seg_img_vis, seg_contours, -1, (0, 0, 255), 1)
        cv2.imwrite(f"{out_dir}/vis_seg/{image_name}.png", seg_img_vis)
        
        # implant
        imp_contours, _ = cv2.findContours(imp_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        imp_img_vis = cv2.cvtColor(imp_img, cv2.COLOR_GRAY2BGR) * 255
        largest_imp_contour = max(imp_contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest_imp_contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(imp_img_vis, [box], -1, (0, 0, 255), 1)
        cv2.imwrite(f"{out_dir}/vis_imp/{image_name}.png", imp_img_vis)
        
        # combine visualize
        cmb_vis = np.zeros_like(imp_img_vis)
        cv2.drawContours(cmb_vis, seg_contours, -1, (255, 0, 0), 1)
        cv2.drawContours(cmb_vis, [box], -1, (0, 0, 255), 1)
        cv2.imwrite(f"{out_dir}/vis_cmb/{image_name}.png", cmb_vis)
        
        # combine modality
        cmb_mod = np.zeros_like(imp_img_vis)
        cv2.drawContours(cmb_mod, [box], -1, (0, 0, 255), -1)
        cmb_mod[:, :, 0] = seg_img * 255
        # cmb_mod[:, :, 2] = imp_img * 255
        cv2.imwrite(f"{out_dir}/vis_cmb_mod/{image_name}.png", cmb_mod)


if __name__ == '__main__':
    
    terheyden(
        seg_dir='datasets/bone_seg/predictTs', 
        implant_dir='datasets/sagittal_plane/coronal', 
        out_dir='datasets/combine_implant_bone'
    )
