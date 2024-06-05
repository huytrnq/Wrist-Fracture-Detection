import os 
import cv2
import numpy as np
from glob import glob

from models.models import Wrist_Fracture_Detection
from models.kernels import HogDescriptor
from utils.dataset import load_yolo_labels
from utils.vis import draw_bboxes
from utils.bboxes import convert_to_yolo_format, save_yolo_labels
from utils.ensemble import combine_bboxes_to_polygon, draw_polygon_on_image, get_boxes_inside_polygon

# Load the trained classifier

if __name__ == '__main__':
    export_results = True
    mode = 'test'
    class_name = 'fracture'
    
    save_path = os.path.join('results', mode, class_name)
    os.makedirs(save_path, exist_ok=True)
    infer_path = 'MLDataset/crop_data/images'
    
    classifier_path = 'haar_training/classifier/cascade.xml'
    haar_cascade = cv2.CascadeClassifier(classifier_path)
    
    hog_descriptor = HogDescriptor()
    wfd = Wrist_Fracture_Detection(model='models/weights/svm_hog2.pkl', feature_extractor=hog_descriptor, window_size=64, step_size=32, heatmap=None)
    
    
    for image_path in glob(os.path.join(infer_path, mode, class_name, '*.png')):
        try:
            image_name = os.path.basename(image_path)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            image_org = image.copy()
            label_path = image_path.replace('images', 'labels').replace('png', 'txt')
            labels = load_yolo_labels(label_path, image.shape, [3])
            
            # Detect features using the HoG classifier
            rois = wfd.predict(image)
            
            # Detect features using the Haar classifier
            detected_regions = haar_cascade.detectMultiScale(image)
            
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            # print(f"Predictions: {rois}")
            # for xyxy in rois:
            #     x1, y1, x2, y2 = xyxy
            #     cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # draw_bboxes(image, labels, (0, 0, 255))
            
            # if export_results:
            #     yolo_boxes = convert_to_yolo_format(image, rois)
            #     save_yolo_labels(yolo_boxes, os.path.join(save_path, image_name.replace('png', 'txt')))
            #     cv2.imwrite(os.path.join(save_path, image_name), image)
            
            polygon = combine_bboxes_to_polygon(rois)
            image = draw_polygon_on_image(image, polygon)
            
            final_boxes = get_boxes_inside_polygon(detected_regions, polygon, type='xywh')
            for (x, y, w, h) in final_boxes:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            cv2.imshow('image', image)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break
            cv2.destroyAllWindows()
            print(f"Predictions: {rois}")
            print(f"Image: {image_path}")
        except Exception as e:
            print(e)
            
    cv2.destroyAllWindows()
    