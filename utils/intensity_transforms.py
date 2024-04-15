from typing import Any
import cv2

class IntensityTransformation:
    def __init__(self, linear_streching=False, hist_eq=False):
        self.linear_streching = linear_streching
        self.hist_eq = hist_eq
        
    def __call__(self,img) -> Any:
        if self.linear_streching:
            return self.linear_strerching(img=img)
        elif self.hist_eq:
            return self.histogram_equalization(img=img)
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
    
    def histogram_equalization(self, img):
        img = self.load_img(img) if isinstance(img, str) else img
        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.equalizeHist(img)
        return img


if __name__ == '__main__':
    img_path = "C://Users\lok20\OneDrive\_Master\MAIA-ERASMUS//2 Semester\Interdiscipilanry Project AIA_ML_DL\GRAZPEDWRI-DX\images_part1//0001_1297860395_01_WRI-L1_M014.png"
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    assert img is not None, "file could not be read, check with os.path.exists()"
    intensity = IntensityTransformation()
    linear_streched_img = intensity.linear_strerching(img)
    his_eq_img = intensity.histogram_equalization(img)
    cv2.imshow('Linear Stretching', his_eq_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()