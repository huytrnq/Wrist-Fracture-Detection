from joblib import load
from joblib import dump

import cv2
import numpy as np
import skimage
from skimage import exposure
from skimage.filters import unsharp_mask
from skimage.filters import threshold_otsu
from skimage.morphology import disk, closing, dilation

from utils.preprocess import sliding_window
from utils.intensity_transforms import histogram_matching


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


# Function to get original coordinates from pooled coordinates
def get_original_coordinates(pooled_coords, pooling_factor):
    """
    Convert coordinates from pooled image to original image coordinates.

    Args:
    pooled_coords (tuple): Coordinates in the pooled image (row1, col1, row2, col2).
    pooling_factor (tuple): Pooling factor (row_factor, col_factor).

    Returns:
    tuple: Coordinates in the original image.
    """
    row1, col1, row2, col2 = pooled_coords
    row_factor, col_factor = pooling_factor
    return row1 * row_factor, col1 * col_factor, row2 * row_factor, col2 * col_factor


def iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1b, y1b, x2b, y2b = box2

    # Calculate intersection
    inter_x1 = max(x1, x1b)
    inter_y1 = max(y1, y1b)
    inter_x2 = min(x2, x2b)
    inter_y2 = min(y2, y2b)
    inter_area = max(0, inter_x2 - inter_x1 + 1) * max(0, inter_y2 - inter_y1 + 1)

    # Calculate union
    box1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    box2_area = (x2b - x1b + 1) * (y2b - y1b + 1)
    union_area = box1_area + box2_area - inter_area

    # Calculate IoU
    return inter_area / union_area


def merge_two_boxes(box1, box2):
    x1 = min(box1[0], box2[0])
    y1 = min(box1[1], box2[1])
    x2 = max(box1[2], box2[2])
    y2 = max(box1[3], box2[3])
    return [x1, y1, x2, y2]


class Wrist_Fracture_Detection:
    def __init__(
        self,
        model=None,
        step_size=64,
        window_size=64,
        pool_size=(4, 4),
        feature_extractor=None,
        heatmap="./heatmap.npy",
    ):
        """Wrist Fracture Detection

        Args:
            model (str, optional): Path to exported model. Defaults to None.
            step_size (int, optional): Stride step. Defaults to 128.
            window_size (int, optional): Sliding window size. Defaults to 256.
            pool_size (tuple, optional): Pooling window size. Defaults to (4, 4).
            feature_extractor (function, optional): Feature extractor. Defaults to None.
            heatmap (str, optional): Path to heatmap. Defaults to './heatmap.npy'.
        """
        self.model = self.load(model) if model else None
        self.step_size = step_size
        self.window_size = window_size
        self.pool_size = pool_size
        self.feature_extractor = feature_extractor
        self.heatmap = heatmap
        self.roi_offset_left = (0,0)

    def get_roi_from_heatmap(self, image):
        """Get region of interest from heatmap

        Args:
            image array): Image

        Returns:
            roi: np.array
        """
        heatmap = np.load(self.heatmap)
        coords = np.column_stack(np.where(heatmap > 0))
        top_left = coords.min(axis=0)
        bottom_right = coords.max(axis=0)
        ## Scale the roi coordinates
        ## order h, w
        scaled_roi_coords = [
            top_left[0] / heatmap.shape[0] * image.shape[0],
            top_left[1] / heatmap.shape[1] * image.shape[1],
            bottom_right[0] / heatmap.shape[0] * image.shape[0],
            bottom_right[1] / heatmap.shape[1] * image.shape[1],
        ]
        roi_image = image[
            int(scaled_roi_coords[0]) : int(scaled_roi_coords[2]),
            int(scaled_roi_coords[1]) : int(scaled_roi_coords[3]),
        ]
        ## Set the offset and change to w, h
        self.roi_offset_left = (int(scaled_roi_coords[1]), int(scaled_roi_coords[0]))
        return roi_image

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

    def fit(self, X, y):
        """Fit the model

        Args:
            X (np.array): Features
            y (np.array): Labels
        """
        self.model.fit(X, y)

    def predict(self, image):
        """Predict

        Args:
            X (np.array): Features

        Returns:
            Predictions: np.array
        """
        rois = []
        scaled_roi = []

        feature_windows = []
        if self.heatmap:
            image = self.get_roi_from_heatmap(image)
        image = self.preprocess(image)
        image = self.padding(image, factor=self.window_size)
        for x, y, window in sliding_window(image, self.step_size, self.window_size):
            if (
                window.shape[0] != self.window_size
                or window.shape[1] != self.window_size
            ):
                continue
            window = skimage.measure.block_reduce(window, (2, 2), np.max)
            features = self.feature_extraction(window)
            feature_windows.append(features)
            rois.append((x, y, x + self.window_size, y + self.window_size))

        results = self.model.predict(feature_windows)
        # probs = self.model.predict_proba(feature_windows)
        ### Threshold the probabilities to get the final results
        # results[probs[:,1] < 0.6] = 0
        ### Get the rois which contain the fracture
        rois = np.array(rois)
        selected_rois = rois[results == 1]
        for roi in selected_rois:
            roi = get_original_coordinates(roi, self.pool_size)
            ## Add the offset to the roi
            roi = (roi[0] + self.roi_offset_left[0], roi[1] + self.roi_offset_left[1], roi[2] + self.roi_offset_left[0], roi[3] + self.roi_offset_left[1])
            scaled_roi.append(roi)
        # return self.merge_boxes(scaled_roi)
        return scaled_roi

    def save(self, filename):
        """Save the model

        Args:
            filename (str): Path to save the model
        """
        export_model(self.model, filename)

    def load(self, filename):
        """Load the model

        Args:
            filename (str): Path to load the model
        """
        return load_model(filename)

    def pooling(self, img, pool_size=(4, 4)):
        """Apply pooling to the image

        Args:
            img (array): Image
            pool_size (tuple, optional): Pooling Window Size. Defaults to 2.
        """
        img = skimage.measure.block_reduce(img, pool_size, np.max)
        return img

    def preprocess(self, img):
        """Preprocess the image

        Args:
            img (np.array): Image

        Returns:
            np.array: Preprocessed image
        """

        outputbitdepth = 8
        intensity_crop = 1
        dilate_num = 4
        if self.pool_size:
            img = self.pooling(img, self.pool_size)
        ### Load mean histogram from the training data
        mean_hist = np.load("mean_hist.npy")
        ### Unsharp masking
        img = (
            ((unsharp_mask(img, radius=2, amount=1)) * 255)
            .clip(0, 255)
            .astype(np.uint8)
        )

        ## Histogram equalization
        img = exposure.rescale_intensity(
            img,
            in_range=(
                np.percentile(img, intensity_crop),
                np.percentile(img, (100 - intensity_crop)),
            ),
        )
        img = exposure.equalize_adapthist(img)

        ## Normalize img
        img = cv2.normalize(
            img,
            dst=None,
            alpha=0,
            beta=int((pow(2, outputbitdepth)) - 1),
            norm_type=cv2.NORM_MINMAX,
        ).astype(np.uint8)

        ## Apply mean histogram
        img = histogram_matching(img, mean_hist)

        # Apply Otsu's thresholding
        thresh = threshold_otsu(img)
        binary_mask = img > thresh

        # Apply multiple dilations
        selem = disk(4)
        dilated_mask = binary_mask
        for _ in range(dilate_num):
            dilated_mask = dilation(dilated_mask, selem)

        # Apply closing to the dilated mask
        closed_mask = closing(dilated_mask, selem)

        # Apply the mask to the original image using a bitwise AND operation
        img = cv2.bitwise_and(img, img, mask=np.uint8(closed_mask * 255))

        return img

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

    def visualize_rois(self, img, rois):
        """Visualize regions of interest

        Args:
            img (np.array): Image
            rois (list): List of regions of interest
        """
        for x, y in rois:
            cv2.rectangle(
                img,
                (x, y),
                (x + self.window_size, y + self.window_size),
                (0, 255, 0),
                2,
            )
        cv2.imshow("Image", img)
        if cv2.waitKey(0) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            import sys

            sys.exit(0)
