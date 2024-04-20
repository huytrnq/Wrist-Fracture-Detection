import os 
import cv2
import numpy as np
from glob import glob

from utils.transforms import Compose
from utils.edge_detection import Canny
from utils.thresholding import ImageThresholding
from utils.intensity_transforms import IntensityTransformation
from utils.preprocess import Preprocessor

if __name__ == '__main__':
    path = '/Users/huytrq/Workspace/unicas/AIA&ML/Wrist-Fracture-Detection/dataset/img/fracture'
    for image_path in glob(os.path.join(path, '*.png')):
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        intensity = IntensityTransformation()
        img_16 = intensity.equalize_histogram_16bit_manual(img)
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        img_8 = intensity.histogram_equalization(img)
        cv2.imshow('16-bit', img_16)
        cv2.imshow('8-bit', img_8)
        cv2.waitKey(0)
        cv2.destroyAllWindows()