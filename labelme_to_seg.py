import json
import os
import shutil
import cv2

import math
import uuid
import random
from typing import Optional

import numpy as np
import numpy.typing as npt
import PIL.Image
import PIL.ImageDraw


def convert_labelme_to_nnunet(raw_file, out_file, random_seed=21):
    os.makedirs(f"{out_file}/imagesTr", exist_ok=True)
    os.makedirs(f"{out_file}/labelsTr", exist_ok=True)
    os.makedirs(f"{out_file}/imagesTs", exist_ok=True)
    os.makedirs(f"{out_file}/labelsTs", exist_ok=True)
    os.makedirs(f"{out_file}/visualTr", exist_ok=True)

    files = [item.path for item in os.scandir(raw_file) if item.is_file()]
    img_files = [f for f in files if f.endswith("png")]
    label_files = [f for f in files if f.endswith("json")]
    data_names = [os.path.splitext(os.path.basename(f))[0] for f in img_files]
    img_files.sort()
    label_files.sort()
    data_names.sort()

    random.seed(random_seed)
    random.shuffle(data_names)
    train_split = data_names[:round(0.8*len(data_names))]
    val_split = data_names[round(0.8*len(data_names)):]
    train_split.sort()
    val_split.sort()
    splits_final = [{"train":train_split,"val":val_split}]
    with open(os.path.join(out_file, "splits_final.json"), 'w', encoding='utf-8') as sf:
        json.dump(splits_final, sf, indent=4)

    for i in range(len(img_files)):
        filename = os.path.split(img_files[i])[-1].split('.')[0]
        print(filename, label_files[i], img_files[i])
        data = json.load(open(label_files[i]))
        lbl, _, lbl_names = labelme_shapes_to_label(cv2.imread(img_files[i]).shape[:2], data['shapes'])

        # make training data
        img = PIL.Image.open(img_files[i]).convert("L")
        img.save(os.path.join(out_file, f"imagesTr", f"{filename}_0000.png"))
        PIL.Image.fromarray(lbl).save(os.path.join(out_file, f"labelsTr", '{}.png'.format(filename)))

        # make validation data
        if filename in val_split :
            img.save(os.path.join(out_file, f"imagesTs", f"{filename}_0000.png"))
            PIL.Image.fromarray(lbl).save(os.path.join(out_file, f"labelsTs", '{}.png'.format(filename)))

        # make visualization
        img = PIL.Image.open(img_files[i]).convert("RGB")
        boost = np.array([120, 0, 0], dtype=np.float32)
        vis = np.array(img) + np.stack([lbl > 0] * 3, axis=-1) * boost
        vis = np.clip(vis, 0, 255).astype(np.uint8)
        vis_img = PIL.Image.fromarray(vis)
        vis_img.save(os.path.join(out_file, "visualTr", f"{filename}_vis.png"))


def shape_to_mask(
    img_shape: tuple[int, ...],
    points: list[list[float]],
    shape_type: Optional[str] = None,
    line_width: int = 10,
    point_size: int = 5,
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
    
    convert_labelme_to_nnunet(
        raw_file='datasets/bone_seg/labelme',
        out_file='datasets/bone_seg'
    )
