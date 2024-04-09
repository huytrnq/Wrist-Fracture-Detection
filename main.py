import os 
import cv2

from utils.preprocess import Preprocessor

if __name__ == '__main__':
    path = '/Users/huytrq/Downloads/Compress/Extracted/folder_structure/supervisely/wrist/img'
    img_size = (256, 256)
    preprocess = Preprocessor(path, img_size)
    for path, img0, img in preprocess:
        print(path)
        cv2.imshow('Image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()