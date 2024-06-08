import numpy as np
import cv2

class Compose:
    def __init__(self, transforms, padding_factor=None):
        self.padding_factor = padding_factor
        self.transforms = transforms

    def __call__(self, image):
        for transform in self.transforms:
            image = transform(image)
        return image
    
    
    def padding(self, image, mode="constant", value=0):
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
        target_h = ((h + self.padding_factor - 1) // self.padding_factor) * self.padding_factor
        target_w = ((w + self.padding_factor - 1) // self.padding_factor) * self.padding_factor

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