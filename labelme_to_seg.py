import json
import os
import shutil
import cv2

import math
import uuid
from typing import Optional

import numpy as np
import numpy.typing as npt
import PIL.Image
import PIL.ImageDraw
import argparse


def get_args_parser():
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--data_path', type=str, default='datasets/test/sagittal_plane/labelme')
    parser.add_argument('--result_path', type=str, default='datasets/test')
    return parser


def convert_labelme_to_mask(args):
    os.makedirs(f"{args.result_path}/images", exist_ok=True)
    os.makedirs(f"{args.result_path}/labels", exist_ok=True)
    os.makedirs(f"{args.result_path}/visual", exist_ok=True)

    files = [item.path for item in os.scandir(args.data_path) if item.is_file()]
    img_files = [f for f in files if f.endswith("png")]
    label_files = [f for f in files if f.endswith("json")]
    data_names = [os.path.splitext(os.path.basename(f))[0] for f in img_files]
    img_files.sort()
    label_files.sort()
    data_names.sort()

    for i in range(len(img_files)):
        filename = os.path.splitext(os.path.basename(img_files[i]))[0].replace("_0000", "")

        # Load image + annotation
        data = json.load(open(label_files[i]))
        lbl, _, _ = labelme_shapes_to_label(cv2.imread(img_files[i]).shape[:2], data['shapes'])

        # Convert to 0–1 binary mask
        lbl = (lbl > 0).astype(np.uint8)

        # Save grayscale image
        img = PIL.Image.open(img_files[i]).convert("L")
        img.save(os.path.join(args.result_path, "images", f"{filename}.png"))

        # Save binary label map
        PIL.Image.fromarray(lbl * 255).save(os.path.join(args.result_path, "labels", f"{filename}.png"))

        # Create visualization overlay
        img_rgb = PIL.Image.open(img_files[i]).convert("RGB")
        img_arr = np.array(img_rgb, dtype=np.float32)
        overlay_color = np.array([255, 0, 0], dtype=np.float32)
        vis = img_arr.copy()
        vis[lbl > 0] = vis[lbl > 0] * 0.5 + overlay_color * 0.5
        vis = np.clip(vis, 0, 255).astype(np.uint8)
        PIL.Image.fromarray(vis).save(os.path.join(args.result_path, "visual", f"{filename}.png"))

        print(f"Saved: {filename}.png")


def shape_to_mask(
    img_shape,
    points,
    shape_type= None,
    line_width = 10,
    point_size= 5,
) -> npt.NDArray[np.bool_]:
    mask = PIL.Image.fromarray(np.zeros(img_shape[:2], dtype=np.uint8))
    draw = PIL.ImageDraw.Draw(mask)
    xy = [tuple(point) for point in points]
    if shape_type == "circle":
        assert len(xy) == 2, "Shape of shape_type=circle must have 2 points"
        (cx, cy), (px, py) = xy
        d = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
        draw.ellipse([cx - d, cy - d, cx + d, cy + d], outline=1, fill=1)
    elif shape_type == "rectangle":
        assert len(xy) == 2, "Shape of shape_type=rectangle must have 2 points"
        draw.rectangle(xy, outline=1, fill=1)  # type: ignore[arg-type]
    elif shape_type == "line":
        assert len(xy) == 2, "Shape of shape_type=line must have 2 points"
        draw.line(xy=xy, fill=1, width=line_width)  # type: ignore[arg-type]
    elif shape_type == "linestrip":
        draw.line(xy=xy, fill=1, width=line_width)  # type: ignore[arg-type]
    elif shape_type == "point":
        assert len(xy) == 1, "Shape of shape_type=point must have 1 points"
        cx, cy = xy[0]
        r = point_size
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=1, fill=1)
    elif shape_type in [None, "polygon"]:
        assert len(xy) > 2, "Polygon must have points more than 2"
        draw.polygon(xy=xy, outline=1, fill=1)  # type: ignore[arg-type]
    else:
        raise ValueError(f"shape_type={shape_type!r} is not supported.")
    return np.array(mask, dtype=bool)


def labelme_shapes_to_label(img_shape, shapes):
    label_name_to_value = {"_background_": 0}
    for shape in shapes:
        label_name = shape["label"]
        if label_name in label_name_to_value:
            label_value = label_name_to_value[label_name]
        else:
            label_value = len(label_name_to_value)
            label_name_to_value[label_name] = label_value
    
    cls = np.zeros(img_shape[:2], dtype=np.int32)
    ins = np.zeros_like(cls)
    instances = []
    for shape in shapes:
        points = shape["points"]
        label = shape["label"]
        group_id = shape.get("group_id")
        if group_id is None:
            group_id = uuid.uuid1()
        shape_type = shape.get("shape_type", None)

        cls_name = label
        instance = (cls_name, group_id)

        if instance not in instances:
            instances.append(instance)
        ins_id = instances.index(instance) + 1
        cls_id = label_name_to_value[cls_name]

        mask: npt.NDArray[np.bool_]
        if shape_type == "mask":
            if not isinstance(shape["mask"], np.ndarray):
                raise ValueError("shape['mask'] must be numpy.ndarray")
            mask = np.zeros(img_shape[:2], dtype=bool)
            (x1, y1), (x2, y2) = np.asarray(points).astype(int)
            mask[y1 : y2 + 1, x1 : x2 + 1] = shape["mask"]
        else:
            mask = shape_to_mask(img_shape[:2], points, shape_type)

        cls[mask] = cls_id
        ins[mask] = ins_id
    
    return cls, ins, label_name_to_value


if __name__ == '__main__':
    parser = argparse.ArgumentParser('', parents=[get_args_parser()])
    args = parser.parse_args()

    convert_labelme_to_mask(args)