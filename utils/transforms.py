import numpy as np
import cv2
import skimage
from skimage import exposure
from skimage.filters import threshold_otsu
from skimage.filters import unsharp_mask
from skimage.morphology import disk, closing, dilation
from skimage.exposure import rescale_intensity

from utils.intensity_transforms import histogram_matching

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        for transform in self.transforms:
            image = transform(image)
        return image

class Padding:
    def __init__(self, factor=64, mode="constant", value=0):
        """Initializes the Padding transform.

        Args:
            factor (int, optional): Factor to which the dimensions should be multiples of. Defaults to 64.
            mode (str, optional): Padding mode. Defaults to "constant".
            value (int, optional): value to fill in padding. Defaults to 0.
        """
        self.factor = factor
        self.mode = mode
        self.value = value

    def __call__(self, image):
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
        target_h = ((h + self.factor - 1) // self.factor) * self.factor
        target_w = ((w + self.factor - 1) // self.factor) * self.factor

        # Calculate the padding amounts
        pad_h = target_h - h
        pad_w = target_w - w

        # Pad the image
        if len(image.shape) == 3:  # Color image
            padded_image = np.pad(
                image,
                ((0, pad_h), (0, pad_w), (0, 0)),
                mode=self.mode,
                constant_values=self.value,
            )
        else:  # Grayscale image
            padded_image = np.pad(
                image, ((0, pad_h), (0, pad_w)), mode=self.mode, constant_values=self.value
            )

        return padded_image
    
class Pooling:
    def __init__(self, kernel_size=(4, 4), mode='max'):
        """Initializes the Pooling transform.

        Args:
            kernel_size (tuple, optional): The kernel size to use for the pooling. Defaults to (4, 4).
            stride (tuple, optional): The stride to use for the pooling. Defaults to (4, 4).
            padding (tuple, optional): The padding to use for the pooling. Defaults to (0, 0).
            mode (str, optional): The mode to use for the pooling. Defaults to 'max'.
        """
        self.kernel_size = kernel_size
        self.mode = mode

    def __call__(self, image):
        if self.mode == 'max':
            return skimage.measure.block_reduce(image, self.kernel_size, np.max)
        elif self.mode == 'mean':
            image = skimage.measure.block_reduce(image, self.kernel_size, np.mean)
            rescaled_matrix = rescale_intensity(image, in_range='image', out_range=(0, 255))
            return rescaled_matrix

class HistogramMatching:
    def __init__(self, mean_hist=None):
        """Initializes the Histogram Matching transform.

        Args:
            mean_hist (str, optional): The path to the mean histogram to use for the histogram matching. Defaults to None.
        """
        self.mean_hist = np.load(mean_hist)

    def __call__(self, image):
        return histogram_matching(image, self.mean_hist)

class HistogramEqualization:
    def __init__(self, intensity_crop=0.1):
        """Initializes the Histogram Equalization transform.

        Args:
            intensity_crop (tuple, optional): The intensity range to use for the histogram equalization. Defaults to (0, 255).
        """
        self.intensity_crop = intensity_crop

    def __call__(self, image):
        image = exposure.rescale_intensity(
            image,
            in_range=(
                np.percentile(image, self.intensity_crop),
                np.percentile(image, (100 - self.intensity_crop)),
            ),
        )
        image = exposure.equalize_adapthist(image)
        return image

class Normalize:
    def __init__(self, outputbitdepth=8):
        self.outputbitdepth = outputbitdepth

    def __call__(self, image):
        return cv2.normalize(
            image,
            dst=None,
            alpha=0,
            beta=int((pow(2, self.outputbitdepth)) - 1),
            norm_type=cv2.NORM_MINMAX,
        ).astype(np.uint8)

class UnsharpMask:
    def __init__(self, radius=1, amount=1):
        """Initializes the Unsharp Mask transform.

        Args:
            radius (int, optional): The radius to use for the unsharp mask. Defaults to 1.
            amount (int, optional): The amount to use for the unsharp mask. Defaults to 1.
        """
        self.radius = radius
        self.amount = amount
    
    def __call__(self, image):
        return (((unsharp_mask(image, radius=self.radius, amount=self.amount)) * 255)
            .clip(0, 255)
            .astype(np.uint8)
        )

class ROI_Isolation:
    def __init__(self, dilation=4, dilation_radius=2):
        self.dilation = dilation
        self.dilation_radius = dilation_radius
    
    def __call__(self, image):
        # Apply Otsu's thresholding
        thresh = threshold_otsu(image)
        binary_mask = image > thresh

        # Create a structuring element based on the dilation radius
        selem = disk(self.dilation_radius)

        # Apply multiple dilations
        dilated_mask = binary_mask
        for _ in range(self.dilation):
            dilated_mask = dilation(dilated_mask, selem)

        # Apply closing to the dilated mask
        closed_mask = closing(dilated_mask, selem)
        
        contours, _ = cv2.findContours(closed_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find the largest contour based on area
        if contours:
            # Sort contours based on contour area
            sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            # Get the two largest contours
            largest_contour = sorted_contours[:1]  # Get the top two largest contours
            
            mask = np.zeros_like(binary_mask, dtype=np.uint8)
            # Fill the mask with the largest contour
            cv2.drawContours(mask, largest_contour, -1, (255), thickness=cv2.FILLED)
            
            # Apply the mask to the original image
            final_image = cv2.bitwise_and(image, image, mask=mask)
        else:
            final_image =  cv2.bitwise_and(image, image, mask=closed_mask)

        return final_image