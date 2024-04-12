import cv2
import numpy as np

class Convolution:
    def __init__(self, kernel_size):
        self.kernel_size = kernel_size

    def __call__(self, image):
        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (self.kernel_size, self.kernel_size), 0)

        # Define the Sobel kernels
        sobel_kernel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

        # Apply the Sobel kernels using filter2D
        edges_x = cv2.filter2D(blurred, -1, sobel_kernel_x)
        edges_y = cv2.filter2D(blurred, -1, sobel_kernel_y)

        # Threshold the gradient magnitude to get binary edge images
        edges_x = cv2.threshold(np.abs(edges_x), 100, 255, cv2.THRESH_BINARY)[1]
        edges_y = cv2.threshold(np.abs(edges_y), 100, 255, cv2.THRESH_BINARY)[1]

        return edges_x, edges_y, blurred
    
    
if __name__ == '__main__':
    path = '/Users/huytrq/Downloads/Compress/Extracted/folder_structure/supervisely/wrist/img/0001_1297860395_01_WRI-L1_M014.png'
    conv = Convolution(kernel_size=5)
    img = cv2.imread(path)
    edges_x, edges_y, blured = conv(img)
    concat_img = np.concatenate((edges_x, edges_y), axis=1)
    cv2.imshow('Blur', blured)
    cv2.imshow('Results', concat_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()