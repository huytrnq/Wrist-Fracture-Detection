from joblib import load
from joblib import dump

import cv2
import numpy as np
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


class Wrist_Fracture_Detection:
    def __init__(self, model=None, step_size=128, window_size=256):
        """Wrist Fracture Detection

        Args:
            model (str, optional): Path to exported model. Defaults to None.
            step_size (int, optional): Stride step. Defaults to 128.
            window_size (int, optional): Sliding window size. Defaults to 256.
        """
        self.model = self.load(model) if model else None
        self.hog = cv2.HOGDescriptor()
        self.step_size = step_size
        self.window_size = window_size

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

        feature_windows = []
        image = self.preprocess(image)
        for x, y, window in sliding_window(image, self.step_size, self.window_size):
            if window.shape[0] != self.window_size or window.shape[1] != self.window_size:
                continue
            features = self.feature_extraction(window)
            feature_windows.append(features)
            # if self.model.predict(features):
            rois.append((x, y))
            
        results = self.model.predict(feature_windows)
        ### Get the rois which contain the fracture
        rois = np.array(rois)
        selected_rois = rois[results == 1]
        return selected_rois

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
        return self.hog.compute(img).ravel()

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
