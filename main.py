import os 
import cv2
import numpy as np
from glob import glob

from models.models import Wrist_Fracture_Detection
from models.kernels import HogDescriptor, CannyDescriptor, AlexNetDescriptor
from utils.dataset import load_yolo_labels
from utils.vis import draw_bboxes
from utils.bboxes import convert_to_yolo_format, save_yolo_labels

if __name__ == '__main__':
    export_results = True
    mode = 'test'
    class_name = 'fracture'
    
    save_path = os.path.join('results', mode, class_name)
    os.makedirs(save_path, exist_ok=True)
    infer_path = 'MLDataset/images'
    
    hog_descriptor = HogDescriptor()
    canny_descriptor = CannyDescriptor()
    alexnet_descriptor = AlexNetDescriptor()
    feature_extractors = [hog_descriptor, alexnet_descriptor, canny_descriptor]
    wfd = Wrist_Fracture_Detection(model='models/weights/lgb_hog_alex_canny.pkl', feature_extractor=feature_extractors, window_size=64, step_size=32)
    
    show = False
    for image_path in glob(os.path.join(infer_path, mode, class_name, '*.png')):
        print('Processing:', image_path)
        # try:
        image_name = os.path.basename(image_path)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        label_path = image_path.replace('images', 'labels').replace('png', 'txt')
        labels = load_yolo_labels(label_path, image.shape, [3])
        
        rois = wfd.predict(image)
        
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        for xyxy in rois:
            x1, y1, x2, y2 = xyxy
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        draw_bboxes(image, labels, (0, 0, 255))
        
        if export_results:
            yolo_boxes = convert_to_yolo_format(image, rois)
            save_yolo_labels(yolo_boxes, os.path.join(save_path, image_name.replace('png', 'txt')))
            cv2.imwrite(os.path.join(save_path, image_name), image)
        
        if show:
            cv2.imshow('image', image)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break
            cv2.destroyAllWindows()
            print(f"Predictions: {rois}")
            print(f"Image: {image_path}")
        # except Exception as e:
        #     print(e)    