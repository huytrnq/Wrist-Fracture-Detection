import os
import cv2
import glob
import matplotlib.pyplot as plt

from utils.metrics import ObjectDetectionMetrics
from utils.dataset import load_yolo_labels

object_detection_metrics = ObjectDetectionMetrics(iou_threshold=0.5) ## ignore confidence score

# Load the trained classifier
classifier_path = './haar_training/opencv-cascade-tracker/data/cascade.xml'
haar_cascade = cv2.CascadeClassifier(classifier_path)

test_folder = 'MLDataset/crop_data/images/test'
save_folder = 'results'

os.makedirs(save_folder, exist_ok=True)


for image_path in glob.glob(os.path.join(test_folder, '**/*.png')):
    # Load and pre-process the image
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    ## load labels
    label_path = image_path.replace('images', 'labels').replace('png', 'txt')
    if os.path.exists(label_path):
        labels = load_yolo_labels(label_path, image.shape, [3])
        for label in labels:
            object_detection_metrics.add_ground_truth(image_path, label[1:], type='xyxy')

    # Detect features using the Haar classifier
    detected_regions = haar_cascade.detectMultiScale(gray_image)

    # Draw bounding boxes around detected regions
    for (x, y, w, h) in detected_regions:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        object_detection_metrics.add_prediction(image_path, (x, y, x + w, y + h))

    # Display the image with bounding boxes
    cv2.imwrite(os.path.join(save_folder, os.path.basename(image_path)), image)

# Calculate the metrics
results = object_detection_metrics.evaluate()
print(f"Precision: {results['precision']:.2f}")
print(f"Recall: {results['recall']:.2f}")
print(f"F1 Score: {results['f1_score']:.2f}")
print(f"mIOU: {results['mIOU']:.2f}")