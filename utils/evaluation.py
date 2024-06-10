import numpy as np
import os
from sklearn.metrics import precision_recall_fscore_support, average_precision_score

import os


def calculate_iou(box1, box2):
    """
    Calculate IoU (Intersection over Union) between two bounding boxes.
    
    Parameters:
    - box1, box2: Bounding boxes in the format [x_center, y_center, width, height].
    
    Returns:
    - iou: IoU value.
    """
    x1_min = box1[0] - box1[2] / 2
    x1_max = box1[0] + box1[2] / 2
    y1_min = box1[1] - box1[3] / 2
    y1_max = box1[1] + box1[3] / 2

    x2_min = box2[0] - box2[2] / 2
    x2_max = box2[0] + box2[2] / 2
    y2_min = box2[1] - box2[3] / 2
    y2_max = box2[1] + box2[3] / 2

    inter_xmin = max(x1_min, x2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymin = max(y1_min, y2_min)
    inter_ymax = min(y1_max, y2_max)

    inter_area = max(0, inter_xmax - inter_xmin) * max(0, inter_ymax - inter_ymin)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)

    iou = inter_area / (box1_area + box2_area - inter_area)
    return iou

def read_labels(folder, use_confidence=False):
    """
    Read labels from a folder.
    
    Parameters:
    - folder: Path to the folder containing label files.
    - use_confidence: Boolean indicating whether the labels include confidence scores.
    
    Returns:
    - labels: List of labels.
    """
    labels = []
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        with open(filepath, 'r') as file:
            for line in file:
                parts = list(map(float, line.strip().split()))
                if use_confidence:
                    labels.append(parts)
                else:
                    labels.append(parts[:5])  # Exclude confidence if not used
    return labels

def match_predictions(true_labels, pred_labels, iou_threshold):
    """
    Match predictions to true labels using IoU threshold.
    
    Parameters:
    - true_labels: List of true bounding boxes and class labels.
    - pred_labels: List of predicted bounding boxes and class labels.
    - iou_threshold: IoU threshold to consider a prediction as a true positive.
    
    Returns:
    - tp: Number of true positives.
    - fp: Number of false positives.
    - fn: Number of false negatives.
    - tn: Number of true negatives.
    """
    tp, fp, fn = 0, 0, 0
    matched_preds = set()

    for true_label in true_labels:
        matched = False
        for i, pred_label in enumerate(pred_labels):
            if true_label[0] == pred_label[0]:
                iou = calculate_iou(true_label[1:], pred_label[1:5])
                if iou >= iou_threshold:
                    tp += 1
                    matched = True
                    matched_preds.add(i)
                    break
        if not matched:
            fn += 1

    fp = len(pred_labels) - len(matched_preds)
    tn = len(true_labels) - tp - fn  # Simplified assumption

    return tp, fp, fn, tn

def calculate_sensitivity_specificity(tp, fp, fn, tn):
    """
    Calculate sensitivity and specificity.
    
    Parameters:
    - tp: Number of true positives.
    - fp: Number of false positives.
    - fn: Number of false negatives.
    - tn: Number of true negatives.
    
    Returns:
    - sensitivity: Sensitivity value.
    - specificity: Specificity value.
    """
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    return sensitivity, specificity

def main(true_labels_folder, pred_labels_folder, iou_threshold=0.2):
    true_labels = read_labels(true_labels_folder)
    pred_labels = read_labels(pred_labels_folder)
    
    tp, fp, fn, tn = match_predictions(true_labels, pred_labels, iou_threshold)
    
    sensitivity, specificity = calculate_sensitivity_specificity(tp, fp, fn, tn)
    
    print(f"Sensitivity: {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")

    
# Example usage:
true_labels_folder = "./results/groundtruth"
pred_labels_folder = "./results/predictions"

main(true_labels_folder, pred_labels_folder, iou_threshold=0.1)
