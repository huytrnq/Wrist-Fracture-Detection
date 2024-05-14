from skimage import feature, color
import numpy as np

class LBP:
    def __init__(self, numPoints, radius, method='default'):
        # Store the number of points and the radius
        self.numPoints = numPoints
        self.radius = radius
        self.method = method

    def get_lbp_image(self, image):
        # Check if the image is already in grayscale
        if len(image.shape) > 2 and image.shape[2] == 3:
            image = color.rgb2gray(image)
        
        # Compute the LBP representation of the image
        lbp = feature.local_binary_pattern(image, self.numPoints, self.radius, method=self.method)
        return lbp

    def histogram(self, lbp_image, num_bins=256):
        # Compute the histogram of the LBP image
        hist, _ = np.histogram(lbp_image.ravel(), bins=num_bins, range=(0, num_bins))
        # Normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-6)
        return hist
