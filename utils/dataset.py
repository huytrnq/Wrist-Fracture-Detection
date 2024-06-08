import os
import json
from glob import glob

import os
import cv2
import skimage
from skimage.io import imread

from pathlib import Path
from glob import glob

IMG_FORMATS = "bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp", "pfm" 

def resize_keep_aspect_ratio(image, target_size):
    """Resize the image while keeping the aspect ratio

    Args:
        image (array): Image to resize.
        target_size (tuple): Target size in the format (width, height).

    Returns:
        array: Resized image.
    """
    h, w = image.shape[:2]
    target_w, target_h = target_size
    if w > h:
        new_w = target_w
        new_h = int(h * new_w / w)
    else:
        new_h = target_h
        new_w = int(w * new_h / h)
    return cv2.resize(image, (new_w, new_h))

def resize_image_and_bboxes(image, bounding_boxes, new_shape):
    """
    Resize the image and scale the bounding boxes accordingly.

    Parameters:
    - image: np.array, the original image.
    - bounding_boxes: list of tuples, each tuple contains (x_min, y_min, x_max, y_max).
    - new_width: int, the desired width of the resized image.
    - new_height: int, the desired height of the resized image.

    Returns:
    - resized_image: np.array, the resized image.
    - scaled_bounding_boxes: list of tuples, the scaled bounding box coordinates.
    """
    # Original size
    original_height, original_width = image.shape[:2]
    new_height, new_width = new_shape

    # Resize the image
    resized_image = cv2.resize(image, (new_width, new_height))

    # Calculate scaling factors
    scale_x = new_width / original_width
    scale_y = new_height / original_height

    # Scale bounding boxes
    scaled_bounding_boxes = []
    for box in bounding_boxes:
        c, x_min, y_min, x_max, y_max = box
        x_min = int(x_min * scale_x)
        y_min = int(y_min * scale_y)
        x_max = int(x_max * scale_x)
        y_max = int(y_max * scale_y)
        scaled_bounding_boxes.append((c, x_min, y_min, x_max, y_max))

    return resized_image, scaled_bounding_boxes


def load_yolo_labels(path, shape, classes=None, normalize=False):
    """Load the YOLO labels from the file

    Args:
        path (str): Path to the YOLO labels.
        shape (tuple): Shape of the image.
        classes (list, optional): List of classes. Defaults to None. fracture = 3.
        normalize (bool, optional): Whether to normalize the labels. Defaults to False.

    Returns:
        list: List of labels in the format [class, x1, y1, x2, y2].
    """
    height, width = shape[:2]
    with open(path) as file:
        lines = file.readlines()
    labels = []
    for line in lines:
        label = line.strip().split()
        c, x, y, w, h = label
        c = int(float(c))
        x = float(x) * width
        y = float(y) * height
        w = float(w) * width
        h = float(h) * height
        if normalize:
            x /= width
            y /= height
            w /= width
            h /= height
        xyxy = [c, x - w / 2, y - h / 2, x + w / 2, y + h / 2]
        if classes:
            if c not in classes:
                continue
        labels.append(xyxy)
    return labels


def adjust_labels_for_pooling(labels, original_shape, pool_size):
    """
    Adjusts labels to account for pooling.

    Args:
    labels (list): List of labels.
    original_shape (tuple): Shape of the original image.
    pool_size (tuple): Size of the pooling window.

    Returns:
    list: Adjusted labels.
    """
    height, width = original_shape[:2]
    scale_x = pool_size[1]
    scale_y = pool_size[0]

    adjusted_labels = []
    for label in labels:
        c, x1, y1, x2, y2 = label
        x1 /= scale_x
        y1 /= scale_y
        x2 /= scale_x
        y2 /= scale_y
        adjusted_labels.append([c, x1, y1, x2, y2])

    return adjusted_labels

class DataLoader:
    def __init__(self, path, img_size, transforms=None) -> None:
        """Initializes the Preprocessor class

        Args:
            path (str): path to the images
            img_size (tuple): working size of the images
        """
        
        self.transforms = transforms
        
        if isinstance(path, str) and Path(path).suffix == '.txt':
            path = Path(path).read_text().splitlines()
        files = []
        for p in sorted(p) if isinstance(path, (list, tuple)) else [path]:
            p = str(Path(p).resolve())
            if "*" in p:
                files.extend(sorted(glob(p, recursive=True))) #glob
            elif os.path.isdir(p):
                files.extend(sorted(glob(os.path.join(p, "*.*")))) #dir
            elif os.path.isfile(p):
                files.append(p)
            else:
                raise FileNotFoundError(f"File not found: {p}")
            
        images = [image for image in files if image.split('.')[-1].lower() in IMG_FORMATS]
        self.files = images
        self.file_count = len(images)
        
    def __len__(self):
        return self.file_count
    
    def __iter__(self):
        """Initializes the iterator by resetting the count and returning the instance"""
        self.count = 0
        return self
    
    def __next__(self):
        if self.count == self.file_count:
            return StopIteration
        path = self.files[self.count]
        img0 = imread(path, cv2.IMREAD_GRAYSCALE)
        self.count += 1
        assert img0 is not None, f"File not found {path}"
        
        if self.transforms:
            img = self.transforms(img0)
        else:
            img = img0
    
        return path, img0, img
    

if __name__ == '__main__':

    # Path to the JSON file
    root_path = './../dataset/'
    ann_folder = 'ann'
    img_folder = 'img'
    txt_folder = 'txt'

    categories = ['fracture', 'normal']

    for cat in categories:
        os.makedirs(os.path.join(root_path, img_folder, cat), exist_ok=True)
        os.makedirs(os.path.join(root_path, ann_folder, cat), exist_ok=True)
        os.makedirs(os.path.join(root_path, txt_folder, cat), exist_ok=True)

    for file_path in glob(os.path.join(root_path, ann_folder, '*.json')):
        image_name = os.path.basename(file_path).split('.')[0] + '.png'
        json_name = os.path.basename(file_path)
        txt_name = os.path.basename(file_path).split('.')[0] + '.txt'
        
        # Load the JSON content
        with open(file_path, 'r') as file:
            data = json.load(file)
        
        for object in data['objects']:
            try:
                if object['classTitle'] == 'fracture':
                    os.rename(os.path.join(root_path, img_folder, image_name), os.path.join(root_path, img_folder, 'fracture', image_name))
                    os.rename(os.path.join(root_path, ann_folder, json_name), os.path.join(root_path, ann_folder, 'fracture', json_name))
                    # os.rename(os.path.join(root_path, txt_folder, txt_name), os.path.join(root_path, txt_folder, 'fracture', txt_name))
                    break
                else:
                    os.rename(os.path.join(root_path, img_folder, image_name), os.path.join(root_path, img_folder, 'normal', image_name))
                    os.rename(os.path.join(root_path, ann_folder, json_name), os.path.join(root_path, ann_folder, 'normal', json_name))
                    # os.rename(os.path.join(root_path, txt_folder, txt_name), os.path.join(root_path, txt_folder, 'normal', txt_name))
            except Exception as e:
                print(e)
