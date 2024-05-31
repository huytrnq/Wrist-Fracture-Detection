import os 
import cv2
import numpy as np
from glob import glob

from models.models import Wrist_Fracture_Detection
from utils.dataset import load_yolo_labels
from utils.vis import draw_bboxes

if __name__ == '__main__':
    
    wfd = Wrist_Fracture_Detection(model='models/weights/svm.pkl')
    
    path = 'MLDataset/crop_data/images/test'
    
    for image_path in glob(os.path.join(path, 'fracture/*.png')):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        label_path = image_path.replace('images', 'labels').replace('png', 'txt')
        labels = load_yolo_labels(label_path, image.shape, [3])
        
        rois = wfd.predict(image)
        
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        for (x, y) in rois:
            cv2.rectangle(image, (x, y), (x + wfd.window_size, y + wfd.window_size), (0, 255, 0), 2)
        draw_bboxes(image, labels, (0, 0, 255))
        cv2.imshow('image', image)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
        cv2.destroyAllWindows()
        print(f"Predictions: {rois}")
        print(f"Image: {image_path}")
        
    cv2.destroyAllWindows()
    