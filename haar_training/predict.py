import os
import cv2
import glob
import matplotlib.pyplot as plt

# Load the trained classifier
classifier_path = 'classifier/24x24/cascade.xml'
haar_cascade = cv2.CascadeClassifier(classifier_path)

test_folder = '/Users/huytrq/Workspace/unicas/AIA&ML/Wrist-Fracture-Detection/MLDataset/test/fracture/'
save_folder = 'results'

os.makedirs(save_folder, exist_ok=True)

for image_path in glob.glob(os.path.join(test_folder, '*.png')):
    # Load and pre-process the image
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect features using the Haar classifier
    detected_regions = haar_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw bounding boxes around detected regions
    for (x, y, w, h) in detected_regions:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the image with bounding boxes
    cv2.imwrite(os.path.join(save_folder, os.path.basename(image_path)), image)