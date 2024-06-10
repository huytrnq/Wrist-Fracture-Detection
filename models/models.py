from joblib import load
from joblib import dump

import cv2
import numpy as np
import skimage
from skimage import exposure
from skimage.filters import unsharp_mask
from skimage.filters import threshold_otsu
from skimage.morphology import disk, closing, dilation

from utils.dataset import sliding_window
from utils.intensity_transforms import histogram_matching
from utils.bboxes import iou, merge_two_boxes, get_original_coordinates


def export_model(model, filename):
    """
    Exports a trained scikit-learn model to a file using joblib.

    Args:
    - model (sklearn.base.BaseEstimator): The trained model to export.
    - filename (str): The path to the file where the model will be saved.
    """
    dump(model, filename)
    print(f"Model saved to {filename}")


def load_model(filename):
    """
    Loads a scikit-learn model from a file using joblib.

    Args:
    - filename (str): The path to the file from which the model will be loaded.

    Returns:
    - model (sklearn.base.BaseEstimator): The loaded scikit-learn model.
    """
    model = load(filename)
    print(f"Model loaded from {filename}")
    return model


class WristFractureDetection:
    def __init__(
        self,
        model_path=None,
        step_size=64,
        window_size=64,
        pool_size=(4, 4),
        feature_extractor=None,
        scaler='./models/weights/scaler.pkl'
    ):
        """Wrist Fracture Detection

        Args:
            model_path (str, optional): Path to exported model. Defaults to None.
            step_size (int, optional): Stride step. Defaults to 128.
            window_size (int, optional): Sliding window size. Defaults to 256.
            pool_size (tuple, optional): Pooling window size. Defaults to (4, 4).
            feature_extractor (function, optional): Feature extractor. Defaults to None.
            scaler (str, optional): Path to scaler. Defaults to './scaler.pkl'.
        """
        self.model = self.load(model_path) if model_path else None
        self.step_size = step_size
        self.window_size = window_size
        self.pool_size = pool_size
        self.feature_extractor = feature_extractor
        self.scaler = load(scaler)


    def padding(self, image, factor=64, mode="constant", value=0):
        """
        Pads an image so that its dimensions are multiples of a given factor.

        Args:
            image (np.ndarray): The input image.
            factor (int): The factor to which the dimensions should be multiples of.
            mode (str): The mode parameter for np.pad.
            value (int): The padding value to use when mode is 'constant'.

        Returns:
            np.ndarray: The padded image.
        """
        # Get the dimensions of the image
        h, w = image.shape[:2]

        # Calculate the target dimensions
        target_h = ((h + factor - 1) // factor) * factor
        target_w = ((w + factor - 1) // factor) * factor

        # Calculate the padding amounts
        pad_h = target_h - h
        pad_w = target_w - w

        # Pad the image
        if len(image.shape) == 3:  # Color image
            padded_image = np.pad(
                image,
                ((0, pad_h), (0, pad_w), (0, 0)),
                mode=mode,
                constant_values=value,
            )
        else:  # Grayscale image
            padded_image = np.pad(
                image, ((0, pad_h), (0, pad_w)), mode=mode, constant_values=value
            )

        return padded_image

    def predict(self, image, offset):
        """Predict

        Args:
            X (np.array): Features

        Returns:
            Predictions: np.array
        """
        rois = []
        scaled_roi = []

        feature_windows = []
        image = self.padding(image, factor=self.window_size)
        for x, y, window in sliding_window(image, self.step_size, self.window_size):
            if (
                window.shape[0] != self.window_size
                or window.shape[1] != self.window_size
            ):
                continue
            zero_percentage = np.mean(window == 0) * 100
            if zero_percentage > 50:
                continue
            window = skimage.measure.block_reduce(window, (2, 2), np.max)
            
            features = self.feature_extraction(window)
            features = self.scaler.transform(features.reshape(1, -1)).ravel()
            feature_windows.append(features)
            rois.append((x, y, x + self.window_size, y + self.window_size))

        if len(feature_windows) > 0:
            results = self.model.predict(feature_windows)
        else:
            results = []
            return results
        
        ### Get the rois which contain the fracture
        rois = np.array(rois)
        selected_rois = rois[results == 1]
        for roi in selected_rois:
            roi = get_original_coordinates(roi, self.pool_size)
            ## Add the offset to the roi
            roi = (roi[0] + offset[0], roi[1] + offset[1], roi[2] + offset[0], roi[3] + offset[1])
            scaled_roi.append(roi)
        # return self.merge_boxes(scaled_roi)
        return scaled_roi

    def load(self, filename):
        """Load the model

        Args:
            filename (str): Path to load the model
        """
        return load_model(filename)


    def feature_extraction(self, img):
        """Feature extraction

        Args:
            img (np.array): Image

        Returns:
            np.array: Features
        """
        if type(self.feature_extractor) is not list:
            return self.feature_extractor(img).ravel()
        else:
            features = []
            for extractor in self.feature_extractor:
                features.append(extractor(img).ravel())
            return np.concatenate(features)

    def merge_boxes(self, boxes, iou_threshold=0.2):
        """
        Merge overlapping bounding boxes.

        Args:
            boxes (numpy.ndarray): Array of bounding boxes of shape (N, 4), where N is the number of boxes.
                                Each box is represented as [x1, y1, x2, y2].
            iou_threshold (float): Threshold for IoU to merge overlapping boxes.

        Returns:
            numpy.ndarray: Array of merged bounding boxes.
        """
        if len(boxes) == 0:
            return np.array([])
        boxes = np.array(boxes)

        # Convert boxes to float if they are not already
        if boxes.dtype.kind != "f":
            boxes = boxes.astype(np.float32)

        merged_boxes = []
        used = np.zeros(len(boxes), dtype=bool)

        for i in range(len(boxes)):
            if used[i]:
                continue
            current_box = boxes[i]
            for j in range(i + 1, len(boxes)):
                if used[j]:
                    continue
                if iou(current_box, boxes[j]) >= iou_threshold:
                    current_box = merge_two_boxes(current_box, boxes[j])
                    used[j] = True
            merged_boxes.append(current_box)
            used[i] = True

        return np.array(merged_boxes).astype(int).tolist()