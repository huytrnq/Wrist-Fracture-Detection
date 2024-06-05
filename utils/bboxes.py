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


def convert_to_yolo_format(image, bounding_boxes):
    """
    Convert bounding boxes to YOLO format.

    Parameters:
    - image: np.array, the image.
    - bounding_boxes: list of tuples, each tuple contains (x_min, y_min, x_max, y_max).

    Returns:
    - yolo_bounding_boxes: list of tuples, the YOLO formatted bounding box coordinates.
    """
    image_height, image_width = image.shape[:2]

    yolo_bounding_boxes = []
    for box in bounding_boxes:
        x_min, y_min, x_max, y_max = box
        # Calculate center coordinates
        x_center = (x_min + x_max) / 2.0
        y_center = (y_min + y_max) / 2.0
        # Calculate width and height
        width = x_max - x_min
        height = y_max - y_min
        # Normalize
        x_center /= image_width
        y_center /= image_height
        width /= image_width
        height /= image_height
        yolo_bounding_boxes.append((x_center, y_center, width, height))

    return yolo_bounding_boxes

def save_yolo_labels(yolo_bounding_boxes, output_path):
    """
    Save YOLO labels to a file.

    Parameters:
    - yolo_bounding_boxes: list of tuples, the YOLO formatted bounding box coordinates.
    - output_path: str, the path to save the YOLO labels.
    """
    with open(output_path, 'w') as f:
        for bbox in yolo_bounding_boxes:
            x_center, y_center, width, height = bbox
            # Here we assume class label is 0 for all bounding boxes
            f.write(f"0 {x_center} {y_center} {width} {height}\n")
            
            
def resize_bounding_box(box, original_size, new_size, format='xyxy'):
    """
    Resize bounding box with respect to the image after resizing.
    
    Parameters:
    - box: tuple, the bounding box coordinates (c, x_min, y_min, x_max, y_max) or (c, x, y, w, h)
    - original_size: tuple, the original image size (width, height)
    - new_size: tuple, the new image size (width, height)
    - format: str, the format of the bounding box ('xyxy' or 'xywh')
    
    Returns:
    - new_box: tuple, the resized bounding box coordinates in the same format as input
    """
    orig_width, orig_height = original_size
    new_width, new_height = new_size

    # Calculate scaling factors
    scale_x = new_width / orig_width
    scale_y = new_height / orig_height

    if format == 'xyxy':
        c, x_min, y_min, x_max, y_max = box
        x_min = int(x_min * scale_x)
        y_min = int(y_min * scale_y)
        x_max = int(x_max * scale_x)
        y_max = int(y_max * scale_y)
        new_box = (c, x_min, y_min, x_max, y_max)

    elif format == 'xywh':
        c, x, y, w, h = box
        x = int(x * scale_x)
        y = int(y * scale_y)
        w = int(w * scale_x)
        h = int(h * scale_y)
        new_box = (c, x, y, w, h)

    else:
        raise ValueError("Invalid format. Supported formats: 'xyxy', 'xywh'")

    return new_box
