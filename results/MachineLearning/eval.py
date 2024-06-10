import os
from typing import List, Tuple
import numpy as np

def calculate_iou(box1: Tuple[float, float, float, float], box2: Tuple[float, float, float, float]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    Each box is represented by a tuple of (x_min, y_min, x_max, y_max).
    """
    x_min1, y_min1, x_max1, y_max1 = box1
    x_min2, y_min2, x_max2, y_max2 = box2

    # Calculate intersection coordinates
    x_min_inter = max(x_min1, x_min2)
    y_min_inter = max(y_min1, y_min2)
    x_max_inter = min(x_max1, x_max2)
    y_max_inter = min(y_max1, y_max2)

    # Calculate intersection area
    inter_area = max(0, x_max_inter - x_min_inter) * max(0, y_max_inter - y_min_inter)

    # Calculate union area
    box1_area = (x_max1 - x_min1) * (y_max1 - y_min1)
    box2_area = (x_max2 - x_min2) * (y_max2 - y_min2)
    union_area = box1_area + box2_area - inter_area

    # Calculate IoU
    # iou = inter_area / union_area if union_area > 0 else 0.0
    iou = inter_area / box1_area
    return iou

def read_boxes_from_file(file_path: str, is_pred: bool = False) -> List[Tuple[int, float, float, float, float, float]]:
    """
    Read boxes from a file. If `is_pred` is True, read predicted boxes with score.
    """
    boxes = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if is_pred:
                class_id, score, x, y, w, h = map(float, parts)
                boxes.append((int(class_id), score, x, y, w, h))
            else:
                class_id, x, y, w, h = map(float, parts)
                boxes.append((int(class_id), x, y, w, h))
    return boxes

def convert_to_xyxy(box: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    """
    Convert [x, y, w, h] to [x_min, y_min, x_max, y_max].
    """
    x, y, w, h = box
    x_min = x - w / 2
    y_min = y - h / 2
    x_max = x + w / 2
    y_max = y + h / 2
    return x_min, y_min, x_max, y_max

def calculate_sensitivity_from_folders(gt_folder: str, pred_folder: str, iou_threshold: float = 0.5) -> float:
    """
    Calculate sensitivity (recall) from YOLO ground truth and predicted boxes in folders.
    """
    TP = 0
    FN = 0
    total_gt = 0
    total_pred = 0

    gt_files = os.listdir(gt_folder)
    for gt_file in gt_files:
        gt_path = os.path.join(gt_folder, gt_file)
        pred_path = os.path.join(pred_folder, gt_file)

        if not os.path.exists(pred_path):
            continue

        gt_boxes = read_boxes_from_file(gt_path)
        pred_boxes = read_boxes_from_file(pred_path, is_pred=True)

        gt_boxes_xyxy = [convert_to_xyxy(box[1:]) for box in gt_boxes]
        pred_boxes_xyxy = [convert_to_xyxy(box[2:]) for box in pred_boxes]
        total_gt += len(gt_boxes_xyxy)
        total_pred += len(pred_boxes_xyxy)
        

        for gt_box in gt_boxes_xyxy:
            match_found = False
            for pred_box in pred_boxes_xyxy:
                iou = calculate_iou(gt_box, pred_box)
                if iou >= iou_threshold:
                    TP += 1
                    match_found = True
                    break
            if not match_found:
                FN += 1

    # Calculate sensitivity
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    print(f"TP: {TP}, FN: {FN}, total_gt: {total_gt}, total_pred: {total_pred}")
    return sensitivity

# Example usage
gt_folder = '../test/groundtruth'
pred_folder = './predictions_wcf/'

sensitivity = calculate_sensitivity_from_folders(gt_folder, pred_folder, iou_threshold=0.01)
print(f"Sensitivity: {sensitivity:.4f}")
