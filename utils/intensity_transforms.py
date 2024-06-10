from typing import Any
<<<<<<< HEAD
=======
import numpy as np
>>>>>>> huytrq
import cv2

class IntensityTransformation:
    def __init__(self, 
                linear_streching=False, 
                hist_eq=False,
                clahe=False,
                clip_limit=2.0,
                tile_grid_size=(8,8)):
        self.linear_streching = linear_streching
        self.hist_eq = hist_eq
        self.clahe = clahe
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        
    def __call__(self,img) -> Any:
        if self.linear_streching:
            return self.linear_strerching(img=img)
        elif self.hist_eq:
            return self.histogram_equalization(img=img)
        elif self.clahe:
            return self.apply_clahe(img=img)
        else:
            print('No intensity transformation method selected')
            return img
    
    def load_img(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        assert img is not None, "file could not be read, check with os.path.exists()"
        return img
        
    def linear_strerching(self, img):
        img = self.load_img(img) if isinstance(img, str) else img
        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        min_intensity = img.min()
        max_intensity = img.max()
        img = (img - min_intensity) / (max_intensity - min_intensity) * 255
        return img
    
    def apply_clahe(self, img):
        # Convert the image to grayscale if it is not already
        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
        # Calculate the clip limit based on the image size
        clip_limit = self.calculate_clip_limit(img)
        
        # Calculate the tile grid size based on the image size
        tile_grid_size = self.calculate_tile_grid_size(img)
        
        # Create a CLAHE object with the calculated parameters
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        img = clahe.apply(img)
        return img
    
    def calculate_clip_limit(self, img):
        # Calculate the clip limit based on the image size
        height, width = img.shape
        clip_limit = max(height, width) / 40.0
        return clip_limit
    
    def calculate_tile_grid_size(self, img):
        # Calculate the tile grid size based on the image size
        height, width = img.shape
        tile_size = min(height, width) // 8
        tile_grid_size = (tile_size, tile_size)
        return tile_grid_size
    
    def histogram_equalization(self, img):
        img = self.load_img(img) if isinstance(img, str) else img
        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.equalizeHist(img)
        return img
<<<<<<< HEAD
=======
    
    def equalize_histogram_16bit_manual(self,img):
        
        # Check if the image is 16-bit
        if img.dtype != np.uint16:
            raise ValueError("Image is not a 16-bit image.")

        # Calculate the histogram
        hist, bins = np.histogram(img.flatten(), 65536, [0, 65536])
        
        # Calculate the cumulative distribution function
        cdf = hist.cumsum()
        cdf_normalized = cdf * float(hist.max()) / cdf.max()

        # Normalize the CDF
        cdf_m = np.ma.masked_equal(cdf, 0)
        cdf_m = (cdf_m - cdf_m.min()) * 65535 / (cdf_m.max() - cdf_m.min())
        cdf = np.ma.filled(cdf_m, 0).astype('uint16')

        # Map the original pixels to equalized values
        img_equalized = cdf[img]

        return img_equalized


def calculate_mean_histogram(image_paths):
    mean_hist = np.zeros(256)
    num_images = len(image_paths)
    
    for path in image_paths:
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            hist = cv2.calcHist([image], [0], None, [256], [0, 256])
            mean_hist += hist[:, 0]
    
    mean_hist /= num_images
    return mean_hist

def histogram_matching(source_image, reference_histogram):
    # Calculate the histogram of the source image
    src_hist, bins = np.histogram(source_image.flatten(), 256, [0, 256])
    src_cdf = src_hist.cumsum()

    # Normalize the CDF
    src_cdf_normalized = src_cdf * (reference_histogram.sum() / src_cdf.max())
    reference_cdf_normalized = reference_histogram.cumsum()

    # Create a lookup table
    lookup_table = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        # Use 'left' to find the first index where the reference CDF exceeds or equals src_cdf_normalized[i]
        lookup_val = np.searchsorted(reference_cdf_normalized, src_cdf_normalized[i], side='left')
        lookup_table[i] = lookup_val

    # Map the source image pixels to the reference histogram
    matched_image = cv2.LUT(source_image, lookup_table)
    return matched_image
    
>>>>>>> huytrq

if __name__ == '__main__':
    img_path = '/Users/huytrq/Downloads/Compress/Extracted/folder_structure/supervisely/wrist/img/0031_1007172623_02_WRI-R1_M009.png'
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    assert img is not None, "file could not be read, check with os.path.exists()"
    intensity = IntensityTransformation()
    linear_streched_img = intensity.linear_strerching(img)
    his_eq_img = intensity.histogram_equalization(img)
    cv2.imshow('Linear Stretching', his_eq_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()