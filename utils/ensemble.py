import cv2
import numpy as np
from shapely.geometry import box
from shapely.ops import unary_union
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as mplPolygon

def combine_bboxes_to_polygon(bboxes):
    """
    Combine bounding boxes to generate a polygon area.

    Args:
        bboxes (numpy.ndarray): Array of bounding boxes of shape (N, 4), where N is the number of boxes.
                                Each box is represented as [x1, y1, x2, y2].

    Returns:
        shapely.geometry.Polygon: The combined polygon area.
    """
    # Convert bounding boxes to shapely boxes
    polygons = [box(bbox[0], bbox[1], bbox[2], bbox[3]) for bbox in bboxes]

    # Combine all the bounding boxes to form a single polygon area
    combined_polygon = unary_union(polygons)

    return combined_polygon



def draw_polygon_on_image(image, combined_polygon):
    """
    Draw the combined polygon on the original image using OpenCV.

    Args:
        image (np.ndarray): The original image.
        combined_polygon (shapely.geometry.Polygon): The combined polygon area.
    """
    # Create a copy of the image to draw on
    image_with_polygon = image.copy()

    # Draw combined polygon on the image
    if combined_polygon.geom_type == 'Polygon':
        pts = np.array(combined_polygon.exterior.coords, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(image_with_polygon, [pts], isClosed=True, color=(255, 0, 0), thickness=2)
    elif combined_polygon.geom_type == 'MultiPolygon':
        for poly in combined_polygon:
            pts = np.array(poly.exterior.coords, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(image_with_polygon, [pts], isClosed=True, color=(255, 0, 0), thickness=2)

    return image_with_polygon


def get_boxes_inside_polygon(boxes, polygon, type='xyxy'):
    """
    Get the bounding boxes that are inside a given polygon.

    Args:
        boxes (numpy.ndarray): Array of bounding boxes of shape (N, 4).
        polygon (shapely.geometry.Polygon): The polygon to check against.
        type (str): The format of the bounding boxes ('xyxy' or 'xywh').

    Returns:
        list: List of bounding boxes that are inside the polygon.
    """
    inside_boxes = []
    for bbox in boxes:
        if type == 'xyxy':
            bbox_polygon = box(bbox[0], bbox[1], bbox[2], bbox[3])
        elif type == 'xywh':
            bbox_polygon = box(bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3])
        else:
            raise ValueError("Invalid bounding box type. Supported types are 'xyxy' and 'xywh'.")
        
        if polygon.contains(bbox_polygon):
            inside_boxes.append(bbox)
    return inside_boxes