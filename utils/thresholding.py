import cv2
import matplotlib.pyplot as plt
import numpy as np

class ImageThresholding:
    def __init__(self, blur=False, 
                blur_kernel=5, 
                use_global_thresholding=False, 
                use_adaptive_mean_thresholding=False, 
                use_adaptive_gaussian_thresholding=False, 
                use_otsu_thresholding=False, 
                use_triangle_thresholding=False,
                thesh_value=100,
                max_value=255,
                block_size=11,
                C=2):
        self.blur = blur
        self.blur_kernel = blur_kernel
        self.img = None
        self.thresh_value = thesh_value
        self.max_value = max_value
        self.block_size = block_size
        self.C = C
        self.current_img_path = None
        self.use_global_thresholding = use_global_thresholding
        self.use_adaptive_mean_thresholding = use_adaptive_mean_thresholding
        self.use_adaptive_gaussian_thresholding = use_adaptive_gaussian_thresholding
        self.use_otsu_thresholding = use_otsu_thresholding
        self.use_triangle_thresholding = use_triangle_thresholding
        
    def __call__(self, img):
        if self.use_global_thresholding:
            return self.global_thresholding(img)
        elif self.use_adaptive_mean_thresholding:
            return self.adaptive_mean_thresholding(img)
        elif self.use_adaptive_gaussian_thresholding:
            return self.adaptive_gaussian_thresholding(img)
        elif self.use_otsu_thresholding:
            return self.otsu_thresholding(img)
        elif self.use_triangle_thresholding:
            return self.triangle_thresholding(img)
        else:
            print('No thresholding method selected')
            return img
    
    def load_img(self, img_path):
        if img_path == self.current_img_path:
            return self.img
        self.current_img_path = img_path
        self.img = cv2.imread(img_path, 0)
        self.img = cv2.medianBlur(self.img, self.blur_kernel) if self.blur else self.img
        return self.img

    def global_thresholding(self, img):
        self.img = self.load_img(img) if type(img) is str else img
        self.img = self.img.astype(np.uint8)
        ret, th1 = cv2.threshold(self.img, self.threshold_value, self.max_value, cv2.THRESH_BINARY)
        return th1

    def adaptive_mean_thresholding(self, img):
        self.img = self.load_img(img) if type(img) is str else img
        self.img = self.img.astype(np.uint8)
        th2 = cv2.adaptiveThreshold(self.img, self.max_value, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, self.block_size, self.C)
        return th2

    def adaptive_gaussian_thresholding(self, img):
        self.img = self.load_img(img) if type(img) is str else img
        self.img = self.img.astype(np.uint8)
        th3 = cv2.adaptiveThreshold(self.img, self.max_value, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, self.block_size, self.C)
        return th3
    
    def otsu_thresholding(self, img):
        self.img = self.load_img(img) if type(img) is str else img
        self.img = self.img.astype(np.uint8)
        _, th1 = cv2.threshold(self.img, 0, self.max_value, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        return th1

    def triangle_thresholding(self, img):
        self.img = self.load_img(img) if type(img) is str else img
        self.img = self.img.astype(np.uint8)
        _, th1 = cv2.threshold(self.img, 0, self.max_value, cv2.THRESH_BINARY+cv2.THRESH_TRIANGLE)
        return th1
    
    def display_images(self, images, titles):
        for i in range(len(images)):
            plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
            plt.title(titles[i])
            plt.xticks([]),plt.yticks([])
        plt.show()
        
if __name__ == '__main__':
    image_path = '/Users/huytrq/Downloads/Compress/Extracted/folder_structure/supervisely/wrist/img/0029_0571570953_01_WRI-L1_M008.png'
    image = ImageThresholding()
    gt_image = image.global_thresholding(image_path)
    am_image = image.adaptive_mean_thresholding(image_path)
    ag_image = image.adaptive_gaussian_thresholding(image_path)
    ot_image = image.otsu_thresholding(image_path)
    tr_image = image.triangle_thresholding(image_path)
    image.display_images([image.img, gt_image, am_image, ag_image, ot_image, tr_image], ['Original Image', 'Global Thresholding (v=100)', 'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding', 'Otsu Thresholding', 'Triangle Thresholding'])    