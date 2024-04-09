import os 
import cv2

from utils.thresholding import ImageThresholding
from utils.intensity_transforms import IntensityTransformation
from utils.preprocess import Preprocessor

if __name__ == '__main__':
    path = '/Users/huytrq/Downloads/Compress/Extracted/folder_structure/supervisely/wrist/img'
    img_size = (256, 256)
    preprocess = Preprocessor(path, img_size, transforms=IntensityTransformation(hist_eq=True))
    for path, img0, img in preprocess:
        print(path)
        if '0001_1297860395_01_WRI-L1_M014' in path:
            cv2.imshow('Image', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            break