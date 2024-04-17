import cv2
import numpy as np
import skimage as ski

class Canny:
    def __init__(self, kernel_size, canny_low, canny_high):
        self.kernel_size = kernel_size
        self.canny_low = canny_low
        self.canny_high = canny_high

    def __call__(self, image):
        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (self.kernel_size, self.kernel_size), 0)
        blurred = ski.filters.unsharp_mask(image, 1, 1)
        blurred = blurred*65535
        blurred = blurred.astype(np.uint16)
        result = cv2.Canny(blurred, self.canny_low, self.canny_high)

        return result, blurred
    
    
if __name__ == '__main__':
    path ="C://Users\lok20\OneDrive\_Master\MAIA-ERASMUS//2 Semester\Interdiscipilanry Project AIA_ML_DL\GRAZPEDWRI-DX\images_part1//0001_1297860395_01_WRI-L1_M014.png"
    conv = Canny(3, 80, 100)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    edges, blured = conv(img)
    cv2.imshow('Blur', blured)
    cv2.imshow('Results', edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()