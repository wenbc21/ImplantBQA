import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import math
import copy
import cv2
import csv
import argparse
from get_stl import *
import gc
# from stl import mesh

from get_dicom import get_dcm_3d_array, window_transform_3d
from scipy.spatial.transform import Rotation as R
import scipy.ndimage
from skimage.transform import rescale
from get_cross_section import get_cross_section

def get_args_parser():
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--data_path', type=str, default='datasets')
    parser.add_argument('--results_path', type=str, default='datasets/sagittal_plane')
    parser.add_argument('--spacing', type=int, default=0.3)

    return parser


def main(args):
    
    cbct_data_path = [item.path for item in os.scandir(f"{args.data_path}/CBCT") if item.is_dir()]
    cbct_data_path.sort()
    stl_data_path = [item.path for item in os.scandir(f"{args.data_path}/STL") if item.is_file()]
    stl_data_path.sort()
    assert len(cbct_data_path) == len(stl_data_path), "data number not aligned!"

    data_id = []
    spacing_dict = {}
    window_dict = {}
    width_dict = {}
    mip_window_dict = {}
    mip_width_dict = {}
    with open(f'{args.data_path}/metadata.csv', mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            id = row['ID'].zfill(3)
            data_id.append(id)
            spacing_dict[id] = float(row['Spacing'])
            window_dict[id] = float(row['Window'])
            width_dict[id] = float(row['Width'])
            mip_window_dict[id] = float(row['MIPWindow'])
            mip_width_dict[id] = float(row['MIPWidth'])

    os.makedirs(f"{args.results_path}/auxiliary", exist_ok=True)
    os.makedirs(f"{args.results_path}/coronal", exist_ok=True)
    os.makedirs(f"{args.results_path}/combined", exist_ok=True)

    for i in range(len(cbct_data_path)):

        cbct_path = cbct_data_path[i]
        stl_path = stl_data_path[i]
        data_name = os.path.split(cbct_path)[-1][:3]

        dicom_dir = list([item.path for item in os.scandir(cbct_path) if item.is_dir()])[0]
        dicom = get_dcm_3d_array(dicom_dir)
        if spacing_dict[data_name] != args.spacing:
            dicom = rescale(dicom, spacing_dict[data_name] / args.spacing, order=1, preserve_range=True)
        dicom_mip = window_transform_3d(dicom, window_width=mip_width_dict[data_name], window_center=mip_window_dict[data_name]).astype(np.uint8)
        dicom = window_transform_3d(dicom, window_width=width_dict[data_name], window_center=window_dict[data_name]).astype(np.uint8)

        cs_upper, cs_lower, cs_front, cs_rear, cs_left, cs_right = get_cross_section(
            dicom_mip, False, data_name, args.results_path, mip_window_dict[data_name], mip_width_dict[data_name])

        implant = get_stl(stl_path, dicom.shape)

        cylinder = np.array(np.where(implant == 1))
        cs_num = np.min(cylinder[0])
        mask_slice = implant[cs_num+5]
        ys, zs = np.where(mask_slice == 1)
        midy = (np.min(ys) + np.max(ys)) // 2
        midz = (np.min(zs) + np.max(zs)) // 2

        cross_section = dicom_mip[cs_num, :, :]
        
        plt.figure(figsize=(10, 10))
        plt.imshow(cross_section, cmap=plt.cm.bone)

        nearest_point = np.array([(cs_left + cs_right) // 2, (cs_rear*2 + cs_front)//3])
        x1, y1 = nearest_point
        x2, y2 = midz, midy
        slope = (y2 - y1) / (x2 - x1 + 1e-8)
        intercept = y2 - slope * x2

        vals_x = np.linspace(0, 512)
        vals_y = slope * vals_x + intercept
        plt.plot(vals_x, vals_y, '--', c='purple', label='Line through points')
        plt.scatter(x2, y2, s=30, c='purple', marker='o')

        # cs_upper, cs_lower, cs_front, cs_rear, cs_left, cs_right
        plt.axvline(cs_left, c='r',ls='-.')
        plt.axvline(cs_right, c='r',ls='-.')
        plt.axhline(cs_front, c='r',ls='-.')
        plt.axhline(cs_rear, c='r',ls='-.')
            
        plt.xlim(0, cross_section.shape[0])
        plt.ylim(cross_section.shape[1], 0)
        plt.savefig(f"{args.results_path}/auxiliary/{data_name}_direction.png")
        plt.close()

        def get_angles(input, direction) :
            # compute angle between two vectors
            dot_product = np.dot(input, direction)
            norm_vec = np.linalg.norm(input)
            norm_D = np.linalg.norm(direction)
            cos_theta = dot_product / (norm_vec * norm_D)
            theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
            return theta

        vec = np.array([midz, midy]) - nearest_point
        angle_rad = get_angles(vec, np.array([1, 0]))

        original_point = np.array([dicom.shape[0] // 2, midz, midy])
        center = np.array(dicom.shape) // 2
        rotation_matrix = R.from_euler('x', angle_rad).as_matrix()

        offset = center - rotation_matrix @ center
        rotated_volume = scipy.ndimage.affine_transform(dicom, rotation_matrix, offset=offset, order=1)
        rotated_implant = scipy.ndimage.affine_transform(implant, rotation_matrix, offset=offset, order=1)
        cylinder_aligned = np.array(np.where(rotated_implant == 1))
        midy = (np.min(cylinder_aligned[1]) + np.max(cylinder_aligned[1])) // 2
        rotated_point = rotation_matrix @ original_point
        rotated_point = np.round(rotated_point).astype(int)

        slice_img = rotated_volume[:, midy, :].astype("uint8")
        cv2.imwrite(os.path.join(args.results_path, "coronal", f"{data_name}_dicom.png"), slice_img[::-1])
        
        slice_img = rotated_implant[:, midy, :].astype("uint8")
        cv2.imwrite(os.path.join(args.results_path, "coronal", f"{data_name}_implant.png"), slice_img[::-1] * 255)

        slice_img = np.clip(rotated_volume[:, midy, :] + rotated_implant[:, midy, :] * 255, 0, 255).astype("uint8")
        cv2.imwrite(os.path.join(args.results_path, "combined", f"{data_name}.png"), slice_img[::-1])

        print(f"NO. {data_name} done! angle: {angle_rad * 180 / 3.14}")
        gc.collect()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('', parents=[get_args_parser()])
    args = parser.parse_args()

    main(args)