import cv2
import numpy as np

def iou(box1, box2):
    """Calculate the Intersection over Union (IoU) of two bounding boxes.

    Args:
        box1 (list): bounding box in format [x1, y1, x2, y2]
        box2 (list): bounding box in format [x1, y1, x2, y2]

    Returns:
        float: value of the IoU for the two boxes.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    iou = intersection / float(box1_area + box2_area - intersection)
    return iou

def calculate_boxA_percentage(boxA, boxB):
    xA1, yA1, xA2, yA2 = boxA
    xB1, yB1, xB2, yB2 = boxB

    # Calculate the coordinates of the intersection rectangle
    xi1 = max(xA1, xB1)
    yi1 = max(yA1, yB1)
    xi2 = min(xA2, xB2)
    yi2 = min(yA2, yB2)

    # Calculate the width and height of the intersection rectangle
    wi = max(0, xi2 - xi1)
    hi = max(0, yi2 - yi1)

    # Calculate the area of intersection rectangle
    intersection_area = wi * hi

    # Calculate the area of Box A
    areaA = (xA2 - xA1) * (yA2 - yA1)

    # Calculate the percentage of Box A that is overlapping with Box B
    if areaA == 0:
        return 0  # Avoid division by zero
    percentage = (intersection_area / areaA) 

    return percentage