import json
import os
import os.path as osp
import PIL.Image
import shutil
import pandas as pd
import random
import cv2
import numpy as np

def result_visualize(raw_file):
    os.makedirs(f"{raw_file}/visualTs", exist_ok=True)

    img_files = [item.path for item in os.scandir(f"{raw_file}/imagesTs") if item.is_file()]
    res_files = [item.path for item in os.scandir(f"{raw_file}/predictTs") if item.is_file() and item.path.endswith("png")]
    img_files.sort()
    res_files.sort()
    assert len(img_files) == len(res_files), "not equal"

    for i in range(len(img_files)):
        print(img_files[i], res_files[i])
        img = cv2.imread(img_files[i])
        res = cv2.imread(res_files[i])
        filename = os.path.split(img_files[i])[-1].split("_")[0]
        
        vis = img.astype(np.float32) + res.astype(np.float32) * (0, 0, 75)
        vis = np.clip(vis, 0, 255).astype(np.uint8)
        cv2.imwrite(f"{raw_file}/visualTs/{filename}_vis.png", vis)

if __name__ == '__main__':
    
    result_visualize(raw_file='datasets/bone_seg')
