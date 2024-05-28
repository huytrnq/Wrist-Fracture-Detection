import os 
import cv2
import numpy as np
from glob import glob

from models.models import Wrist_Fracture_Detection

if __name__ == '__main__':
    
    wfd = Wrist_Fracture_Detection(model='models/weights/svm.pkl', visualize=True)
    
    path = 'MLDataset/crop_data/images/test'
    
    for image_path in glob(os.path.join(path, '**/*.png')):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        predictions = wfd.predict(image)
        print(f"Predictions: {predictions}")
        print(f"Image: {image_path}")
        print()
    