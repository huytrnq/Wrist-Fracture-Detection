import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

# Helper function to load YOLO format data
def load_yolo_data(file_path, with_scores=True):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    boxes = []
    for line in lines:
        parts = line.strip().split()
        class_id = int(parts[0])
        if with_scores:
            score = float(parts[1])
            center_x = float(parts[2])
            center_y = float(parts[3])
            width = float(parts[4])
            height = float(parts[5])
            boxes.append((class_id, score, center_x, center_y, width, height))
        else:
            center_x = float(parts[1])
            center_y = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            boxes.append((class_id, center_x, center_y, width, height))
    return boxes

# Function to compute IoU (Intersection over Union)
def compute_iou(box1, box2):
    def to_corners(box, with_scores=True):
        if with_scores:
            x, y, w, h = box[2:]
        else:
            x, y, w, h = box[1:]
        return x - w / 2, y - h / 2, x + w / 2, y + h / 2

    x1_min, y1_min, x1_max, y1_max = to_corners(box1)
    x2_min, y2_min, x2_max, y2_max = to_corners(box2, with_scores=False)

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0

# Compute true positives, false positives, and false negatives
def compute_metrics(predictions, ground_truths, iou_threshold=0.55):
    y_true = []
    y_scores = []

    for image_id, pred_boxes in predictions.items():
        gt_boxes = ground_truths.get(image_id, [])
        matched_gt = []

        for pred in pred_boxes:
            class_id, score, px, py, pw, ph = pred
            found_match = False

            for gt in gt_boxes:
                gt_class_id, gx, gy, gw, gh = gt
                if class_id != gt_class_id:
                    continue

                iou = compute_iou(pred, gt)
                if iou >= iou_threshold:
                    found_match = True
                    matched_gt.append(gt)
                    break

            if found_match:
                y_true.append(1)
            else:
                y_true.append(0)
            y_scores.append(score)

        for gt in gt_boxes:
            if gt not in matched_gt:
                y_true.append(1)
                y_scores.append(0)

    return y_true, y_scores

# Directories containing prediction folders and ground truth data
prediction_folders = ['./gelan_c_c_640_val_evolution_lr01/txt', 
                    './gelan_c_c_640_val_lr01/txt',
                    './yolov9_c_c_640_val_lr001/txt',]
                    # '/Users/huytrq/Workspace/unicas/AIA&ML/Wrist-Fracture-Detection/results/predictions_wcf']
                    # './yolov9_e_c_640_val_60epochs_lr001/txt']  # Add more directories as needed
ground_truth_dir = '/Users/huytrq/Workspace/unicas/AIA&ML/Wrist-Fracture-Detection/results/test/groundtruth'  # Directory containing ground truth txt files

ground_truths = {}
for filename in os.listdir(ground_truth_dir):
    image_id = filename.split('.')[0]
    ground_truths[image_id] = load_yolo_data(os.path.join(ground_truth_dir, filename), with_scores=False)

plt.figure(figsize=(12, 10))  # Change figure size here


for pred_dir in prediction_folders:
    predictions = {}
    for filename in os.listdir(pred_dir):
        image_id = filename.split('.')[0]
        predictions[image_id] = load_yolo_data(os.path.join(pred_dir, filename), with_scores=True)

    y_true, y_scores = compute_metrics(predictions, ground_truths)

    # Calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    
    ap = average_precision_score(y_true, y_scores)

    # Plot the precision-recall curve
    plt.plot(recall, precision, marker='.', label=f'{os.path.basename(pred_dir.split("/")[1])} (mAP: {ap:.2f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.xlim([0.0, 1.0])
plt.ylim([0.7, 1.0])
plt.legend()
plt.show()
