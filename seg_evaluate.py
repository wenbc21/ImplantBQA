import os
import csv
import cv2
import numpy as np
import argparse


def get_args_parser():
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--label_path', type=str, default='datasets/inference/labels')
    parser.add_argument('--predict_path', type=str, default='datasets/inference/sagittal_plane/predict')
    parser.add_argument('--result_path', type=str, default='results/inference')
    return parser


def seg_evaluate(args):
    # input
    label_files = [item.path for item in os.scandir(args.label_path) if item.is_file()]
    predict_files = [item.path for item in os.scandir(args.predict_path) if item.is_file()]
    label_files.sort()
    predict_files.sort()
    
    # output
    os.makedirs(args.result_path, exist_ok=True)
    headers = ["Name", "Dice", "IoU"]
    csv_file = os.path.join(args.result_path, 'seg_evaluate.csv')
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)

    # evaluate
    for i in range(len(label_files)):
        # load images
        label_file = label_files[i]
        predict_file = predict_files[i]
        image_name = os.path.split(label_file)[-1].split('.')[0]
        
        label_img = cv2.imread(label_file, cv2.IMREAD_GRAYSCALE)
        predict_img = cv2.imread(predict_file, cv2.IMREAD_GRAYSCALE)
        label_img[label_img > 0] = 1
        predict_img[predict_img > 0] = 1
        
        intersection = np.logical_and(label_img, predict_img).sum()
        union = np.logical_or(label_img, predict_img).sum()
        dice = (2.0 * intersection) / (label_img.sum() + predict_img.sum() + 1e-6)
        iou = intersection / (union + 1e-6)
        
        # save
        metrics = [
            image_name, 
            f"{dice:.4f}", 
            f"{iou:.4f}"
        ]
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(metrics)
        print(image_name, dice, iou)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('', parents=[get_args_parser()])
    args = parser.parse_args()

    seg_evaluate(args)