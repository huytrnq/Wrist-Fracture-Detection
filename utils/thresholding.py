import cv2
import matplotlib.pyplot as plt

class ImageThresholding:
    def __init__(self, blur=False, 
                blur_kernel=5, 
                global_thresholding=False, 
                adaptive_mean_thresholding=False, 
                adaptive_gaussian_thresholding=False, 
                otsu_thresholding=False, 
                triangle_thresholding=False):
        self.blur = blur
        self.blur_kernel = blur_kernel
        self.img = None
        self.current_img_path = None
        self.global_thresholding = global_thresholding
        self.adaptive_mean_thresholding = adaptive_mean_thresholding
        self.adaptive_gaussian_thresholding = adaptive_gaussian_thresholding
        self.otsu_thresholding = otsu_thresholding
        self.triangle_thresholding = triangle_thresholding
        
    def __call__(self, img):
        if self.global_thresholding:
            return self.global_thresholding(img)
        elif self.adaptive_mean_thresholding:
            return self.adaptive_mean_thresholding(img)
        elif self.adaptive_gaussian_thresholding:
            return self.adaptive_gaussian_thresholding(img)
        elif self.otsu_thresholding:
            return self.otsu_thresholding(img)
        elif self.triangle_thresholding:
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

    def global_thresholding(self, img, threshold_value=100, max_value=255):
        self.img = self.load_img(img) if type(img) is str else img
        ret, th1 = cv2.threshold(self.img, threshold_value, max_value, cv2.THRESH_BINARY)
        return th1

    def adaptive_mean_thresholding(self, img, max_value=255, block_size=11, C=2):
        self.img = self.load_img(img) if type(img) is str else img
        th2 = cv2.adaptiveThreshold(self.img, max_value, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, C)
        return th2

    def adaptive_gaussian_thresholding(self, img, max_value=255, block_size=11, C=2):
        self.img = self.load_img(img) if type(img) is str else img
        th3 = cv2.adaptiveThreshold(self.img, max_value, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, C)
        return th3
    
    def otsu_thresholding(self, img, max_value=255):
        self.img = self.load_img(img) if type(img) is str else img
        _, th1 = cv2.threshold(self.img, 0, max_value, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        return th1

    def triangle_thresholding(self, img, max_value=255):
        self.img = self.load_img(img) if type(img) is str else img
        _, th1 = cv2.threshold(self.img, 0, max_value, cv2.THRESH_BINARY+cv2.THRESH_TRIANGLE)
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