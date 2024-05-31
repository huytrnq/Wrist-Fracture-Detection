import os 
import cv2
import numpy as np
from glob import glob

from models.models import Wrist_Fracture_Detection
from models.kernels import HogDescriptor
from utils.dataset import load_yolo_labels
from utils.vis import draw_bboxes

if __name__ == '__main__':
    
    hog_descriptor = HogDescriptor()
    wfd = Wrist_Fracture_Detection(model='models/weights/xgb_hog.pkl', feature_extractor=hog_descriptor, window_size=64, step_size=32)
    
    path = 'MLDataset/crop_data/images/test'
    
    for image_path in glob(os.path.join(path, 'fracture/*.png')):
        try:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            label_path = image_path.replace('images', 'labels').replace('png', 'txt')
            labels = load_yolo_labels(label_path, image.shape, [3])
            
            rois = wfd.predict(image)
            
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            print(f"Predictions: {rois}")
            for xyxy in rois:
                x1, y1, x2, y2 = xyxy
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            draw_bboxes(image, labels, (0, 0, 255))
            cv2.imshow('image', image)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break
            cv2.destroyAllWindows()
            print(f"Predictions: {rois}")
            print(f"Image: {image_path}")
        except Exception as e:
            print(e)
            
    cv2.destroyAllWindows()
    