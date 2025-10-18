import os
import csv
import cv2
import numpy as np
import argparse


def get_args_parser():
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--data_path', type=str, default='datasets/inference')
    parser.add_argument('--result_path', type=str, default='results/inference')
    return parser


def quantify(args):
    # input
    img_files = [item.path for item in os.scandir(f"{args.data_path}/images") if item.is_file()]
    seg_files = [item.path for item in os.scandir(f"{args.data_path}/labels") if item.is_file()]
    implant_files = [item.path for item in os.scandir(f"{args.data_path}/sagittal_plane/implant") if item.is_file()]
    img_files.sort()
    seg_files.sort()
    implant_files.sort()
    
    # output
    headers = [
        "Name", 
        "CEJ-Palatal-0", "CEJ-Palatal-2", "CEJ-Palatal-4", 
        "Apical-Palatal-0", "Apical-Palatal-2", "Apical-Palatal-4",
        "CEJ-Labial-0", "CEJ-Labial-2", "CEJ-Labial-4", 
        "Apical-Labial-0", "Apical-Labial-2", "Apical-Labial-4",
        "Basal", "Palatal-Resorption", "Labial-Resorption",
        "Palatal-Length", "Labial-Length"
    ]
    csv_file = os.path.join(args.result_path, 'quantify.csv')
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
    os.makedirs(f"{args.result_path}/visualization", exist_ok=True)

    # quantify
    for i in range(len(seg_files)):
        # load images
        img_file = img_files[i]
        seg_file = seg_files[i]
        implant_file = implant_files[i]
        image_name = os.path.split(img_file)[-1].split('.')[0]
        img_qnt = {}
        
        ori_img = cv2.imread(img_file)
        seg_img = cv2.imread(seg_file, cv2.IMREAD_GRAYSCALE)
        imp_img = cv2.imread(implant_file, cv2.IMREAD_GRAYSCALE)
        imp_img[imp_img == 255] = 1
        img_vis = ori_img.copy()
        
        # bone seg
        contours, _ = cv2.findContours(seg_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 50]
        red_mask = np.zeros_like(img_vis)
        cv2.drawContours(red_mask, filtered_contours, -1, (0, 0, 75), thickness=cv2.FILLED)
        img_vis = cv2.addWeighted(img_vis, 1, red_mask, 0.3, 0)
        cv2.drawContours(img_vis, filtered_contours, -1, (80, 40, 200), 1)
        
        # implant seg
        imp_contours, _ = cv2.findContours(imp_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_imp_contour = max(imp_contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest_imp_contour)
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        cv2.fillPoly(img_vis, [box], color=(255, 255, 255))
        
        # closest contour
        (center_x, center_y), (w, h), angle = rect
        implant_center = (int(center_x), int(center_y))
        closest_contour = None
        min_distance = float('inf')
        for contour in filtered_contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                contour_center_x = int(M["m10"] / M["m00"])
                contour_center_y = int(M["m01"] / M["m00"])
                distance = np.sqrt((contour_center_x - implant_center[0])**2 + 
                                (contour_center_y - implant_center[1])**2)
                if distance < min_distance:
                    min_distance = distance
                    closest_contour = contour
        bone_mask = np.zeros(img_vis.shape[:2], dtype=np.uint8)
        cv2.drawContours(bone_mask, [closest_contour], contourIdx=-1, color=255, thickness=-1)
        
        # bbox
        sorted_box = sorted(box, key=lambda p: p[0])
        left_points = sorted_box[:2]
        right_points = sorted_box[2:]
        left_points_sorted = sorted(left_points, key=lambda p: p[1])
        top_right, top_left = left_points_sorted[0], left_points_sorted[1]
        right_points_sorted = sorted(right_points, key=lambda p: p[1])
        bottom_right, bottom_left = right_points_sorted[0], right_points_sorted[1]
        
        # axis
        long_axis_vector = np.array([top_left[0] - bottom_left[0], top_left[1] - bottom_left[1]], dtype=float)
        long_axis_unit = long_axis_vector / np.linalg.norm(long_axis_vector)
        short_axis_vector = np.array([top_right[0] - top_left[0], top_right[1] - top_left[1]], dtype=float)
        short_axis_unit = short_axis_vector / np.linalg.norm(short_axis_vector)

        # Basal
        top_center = ((top_left[0] + top_right[0]) // 2, (top_left[1] + top_right[1]) // 2)
        bottom_center = ((bottom_left[0] + bottom_right[0]) // 2, (bottom_left[1] + bottom_right[1]) // 2)
        up_point = find_intersection_with_mask(bottom_center, top_center, long_axis_unit, bone_mask)
        basal_dist = np.linalg.norm(np.array(top_center) - np.array(up_point))
        img_qnt["Basal"] = basal_dist * 0.3
        cv2.line(img_vis, bottom_center, up_point, (200, 80, 40), 1)
        
        # Apical
        for i in [0, 2, 4]:
            t = i / 0.3
            top_center_x = int((top_left[0] + top_right[0]) / 2 - t * long_axis_unit[0])
            top_center_y = int((top_left[1] + top_right[1]) / 2 - t * long_axis_unit[1])
            top_center = (top_center_x, top_center_y)
            
            left_bound = (int(top_left[0] - t * long_axis_unit[0]), int(top_left[1] - t * long_axis_unit[1]))
            left_point = find_intersection_with_mask(top_center, left_bound, -short_axis_unit, bone_mask)
            left_dist = np.linalg.norm(np.array(left_bound) - np.array(left_point))
            img_qnt[f"Apical-Palatal-{i}"] = left_dist * 0.3
            cv2.line(img_vis, top_center, left_point, (200, 80, 40), 1)
            
            right_bound = (int(top_right[0] - t * long_axis_unit[0]), int(top_right[1] - t * long_axis_unit[1]))
            right_point = find_intersection_with_mask(top_center, right_bound, short_axis_unit, bone_mask)
            right_dist = np.linalg.norm(np.array(right_bound) - np.array(right_point))
            img_qnt[f"Apical-Labial-{i}"] = right_dist * 0.3
            cv2.line(img_vis, top_center, right_point, (200, 80, 40), 1)
        
        # CEJ
        for i in [0, 2, 4]:
            t = i / 0.3
            bottom_center_x = int((bottom_left[0] + bottom_right[0]) / 2 + t * long_axis_unit[0])
            bottom_center_y = int((bottom_left[1] + bottom_right[1]) / 2 + t * long_axis_unit[1])
            bottom_center = (bottom_center_x, bottom_center_y)
            
            left_bound = (int(bottom_left[0] + t * long_axis_unit[0]), int(bottom_left[1] + t * long_axis_unit[1]))
            left_point = find_intersection_with_mask(bottom_center, left_bound, -short_axis_unit, bone_mask)
            left_dist = np.linalg.norm(np.array(left_bound) - np.array(left_point))
            img_qnt[f"CEJ-Palatal-{i}"] = left_dist * 0.3
            cv2.line(img_vis, bottom_center, left_point, (200, 80, 40), 1)
            
            right_bound = (int(bottom_right[0] + t * long_axis_unit[0]), int(bottom_right[1] + t * long_axis_unit[1]))
            right_point = find_intersection_with_mask(bottom_center, right_bound, short_axis_unit, bone_mask)
            right_dist = np.linalg.norm(np.array(right_bound) - np.array(right_point))
            img_qnt[f"CEJ-Labial-{i}"] = right_dist * 0.3
            cv2.line(img_vis, bottom_center, right_point, (200, 80, 40), 1)
        
        # Resorption
        left_intersection = get_intersections((top_left, bottom_left), bone_mask, max_gap=15)
        left_length = np.linalg.norm(top_left - bottom_left)
        left_dist = left_length
        if left_intersection :
            left_dist -= np.linalg.norm(top_left - np.array(left_intersection[1]))
            cv2.line(img_vis, top_left, left_intersection[1], (60, 240, 60), 1)
        img_qnt["Palatal-Resorption"] = left_dist * 0.3
        img_qnt["Palatal-Length"] = left_length * 0.3
        
        right_intersection = get_intersections((top_right, bottom_right), bone_mask, max_gap=15)
        right_length = np.linalg.norm(top_right - bottom_right)
        right_dist = right_length
        if right_intersection :
            right_dist -= np.linalg.norm(top_right - np.array(right_intersection[1]))
            cv2.line(img_vis, top_right, right_intersection[1], (60, 240, 60), 1)
        img_qnt["Labial-Resorption"] = right_dist * 0.3
        img_qnt["Labial-Length"] = right_length * 0.3
        
        # save
        metrics = [
            f"{img_qnt["CEJ-Palatal-0"]:.4f}", f"{img_qnt["CEJ-Palatal-2"]:.4f}", f"{img_qnt["CEJ-Palatal-4"]:.4f}", 
            f"{img_qnt["Apical-Palatal-0"]:.4f}", f"{img_qnt["Apical-Palatal-2"]:.4f}", f"{img_qnt["Apical-Palatal-4"]:.4f}", 
            f"{img_qnt["CEJ-Labial-0"]:.4f}", f"{img_qnt["CEJ-Labial-2"]:.4f}", f"{img_qnt["CEJ-Labial-4"]:.4f}", 
            f"{img_qnt["Apical-Labial-0"]:.4f}", f"{img_qnt["Apical-Labial-2"]:.4f}", f"{img_qnt["Apical-Labial-4"]:.4f}", 
            f"{img_qnt["Basal"]:.4f}", f"{img_qnt["Palatal-Resorption"]:.4f}", f"{img_qnt["Labial-Resorption"]:.4f}", 
            f"{img_qnt["Palatal-Length"]:.4f}", f"{img_qnt["Labial-Length"]:.4f}", 
        ]
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([image_name] + metrics)
        cv2.imwrite(f"{args.result_path}/visualization/{image_name}.png", img_vis)
        print(image_name, "done!")


def get_intersections(edge, bone_mask, max_gap=5):
    intersections = []
    p1, p2 = edge
    num_samples = int(np.linalg.norm(p1 - p2))
    xs = np.linspace(p1[0], p2[0], num_samples).astype(int)
    ys = np.linspace(p1[1], p2[1], num_samples).astype(int)

    gap_counter = 0  # Track consecutive empty pixels
    for x, y in zip(xs, ys):
        if 0 <= x < bone_mask.shape[1] and 0 <= y < bone_mask.shape[0]:
            if bone_mask[y, x] > 0:
                intersections.append((x, y))
                gap_counter = 0  # Reset gap counter on intersection
            else:
                gap_counter += 1
                if gap_counter > max_gap:
                    break  # Stop if gap is too large
    
    if len(intersections) == 0:
        return None
    else :
        return [intersections[0], intersections[-1]]


def find_intersection_with_mask(start_point, start_bound, direction_vector, bone_mask):
    h, w = bone_mask.shape[:2]
    x, y = start_point
    dx, dy = direction_vector

    norm = np.hypot(dx, dy)
    if norm == 0:
        return start_bound
    dx, dy = dx / norm, dy / norm

    last_inside = start_bound
    t = np.hypot(start_bound[0] - x, start_bound[1] - y)
    while True:
        xi = int(round(x + t * dx))
        yi = int(round(y + t * dy))
        if xi < 0 or yi < 0 or xi >= w or yi >= h:
            break
        if bone_mask[yi, xi] > 0:
            last_inside = (xi, yi)
        t += 1

    return last_inside


if __name__ == '__main__':
    parser = argparse.ArgumentParser('', parents=[get_args_parser()])
    args = parser.parse_args()

    quantify(args)