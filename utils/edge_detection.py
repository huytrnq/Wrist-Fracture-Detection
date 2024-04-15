import cv2
import numpy as np

class Canny:
    def __init__(self, kernel_size):
        self.kernel_size = kernel_size

    def __call__(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image
        blurred = cv2.GaussianBlur(gray, (self.kernel_size, self.kernel_size), 0)
        _, edges = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        eroded = cv2.erode(edges, None, iterations=2)
        return eroded

    
if __name__ == '__main__':
    path = '/Users/huytrq/Downloads/Compress/Extracted/folder_structure/supervisely/wrist/img/0001_1297860395_01_WRI-L1_M014.png'
    conv = Canny()
    img = cv2.imread(path)
    edges, blured = conv(img)
    cv2.imshow('Blur', blured)
    cv2.imshow('Results', edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()