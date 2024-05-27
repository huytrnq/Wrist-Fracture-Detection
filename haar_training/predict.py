import os
import cv2
import glob
import matplotlib.pyplot as plt

# Load the trained classifier
classifier_path = '/Users/huytrq/Workspace/unicas/AIA&ML/Wrist-Fracture-Detection/haar_training/classifier/24x24/cascade.xml'
haar_cascade = cv2.CascadeClassifier(classifier_path)

test_folder = '/Users/huytrq/Workspace/unicas/AIA&ML/Wrist-Fracture-Detection/MLDataset/images/test/normal'
save_folder = 'results'

os.makedirs(save_folder, exist_ok=True)

y_true = []
y_pred = []

for image_path in glob.glob(os.path.join(test_folder, '*.png')):
    # Load and pre-process the image
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect features using the Haar classifier
    detected_regions = haar_cascade.detectMultiScale(gray_image)

    # Draw bounding boxes around detected regions
    for (x, y, w, h) in detected_regions:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the image with bounding boxes
    cv2.imwrite(os.path.join(save_folder, os.path.basename(image_path)), image)