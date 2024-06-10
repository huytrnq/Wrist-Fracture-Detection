import os
import cv2
import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa
from glob import glob

def load_yolo_labels(label_path):
    labels = []
    with open(label_path, 'r') as file:
        for line in file:
            class_id, x_center, y_center, width, height = map(float, line.strip().split())
            labels.append([class_id, x_center, y_center, width, height])
    return labels

def save_yolo_labels(label_path, labels):
    with open(label_path, 'w') as file:
        for label in labels:
            file.write(' '.join(map(str, label)) + '\n')

def augment_image_and_labels(image, bbs, seq):
    image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)
    return image_aug, bbs_aug

def create_bounding_boxes(labels, image_shape):
    bbs = []
    for label in labels:
        class_id, x_center, y_center, width, height = label
        x_center *= image_shape[1]
        y_center *= image_shape[0]
        width *= image_shape[1]
        height *= image_shape[0]
        x_min = int(x_center - width / 2)
        y_min = int(y_center - height / 2)
        x_max = int(x_center + width / 2)
        y_max = int(y_center + height / 2)
        bbs.append(ia.BoundingBox(x1=x_min, y1=y_min, x2=x_max, y2=y_max, label=class_id))
    return bbs

def update_labels_from_bounding_boxes(bbs, image_shape):
    labels = []
    for bb in bbs:
        x_center = (bb.x1 + bb.x2) / 2 / image_shape[1]
        y_center = (bb.y1 + bb.y2) / 2 / image_shape[0]
        width = (bb.x2 - bb.x1) / image_shape[1]
        height = (bb.y2 - bb.y1) / image_shape[0]
        labels.append([bb.label, x_center, y_center, width, height])
    return labels

def generate_augmented_data(image_folder, label_folder, output_image_folder, output_label_folder, num_augmented=5):
    if not os.path.exists(output_image_folder):
        os.makedirs(output_image_folder)
    if not os.path.exists(output_label_folder):
        os.makedirs(output_label_folder)
    
    seq = iaa.Sequential([
        iaa.Fliplr(0.5),  # horizontal flips
        iaa.Affine(scale=(0.8, 1.2), rotate=(-20, 20), translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}),  # scaling and rotation
        iaa.Multiply((0.8, 1.2)),  # change brightness
    ])
    
    image_files = glob(os.path.join(image_folder, '*.png'))  # Adjust the extension as needed
    
    for image_path in image_files:
        label_path = os.path.join(label_folder, os.path.basename(image_path).replace('.png', '.txt'))
        
        image = cv2.imread(image_path)
        labels = []
        bbs = []
        if os.path.exists(label_path):
            labels = load_yolo_labels(label_path)
            bbs = create_bounding_boxes(labels, image.shape)
        
        for i in range(num_augmented):
            image_aug, bbs_aug = augment_image_and_labels(image, bbs, seq)
            
            base_name = os.path.basename(image_path).replace('.png', f'_aug_{i}.png')
            output_image_path = os.path.join(output_image_folder, base_name)
            cv2.imwrite(output_image_path, image_aug)
            if len(labels) > 0:
                augmented_labels = update_labels_from_bounding_boxes(bbs_aug, image.shape)
                output_label_path = os.path.join(output_label_folder, base_name.replace('.png', '.txt'))
                save_yolo_labels(output_label_path, augmented_labels)

if __name__ == '__main__':
    # Example usage
    image_folder = '/Users/huytrq/Workspace/unicas/AIA&ML/Wrist-Fracture-Detection/MLDataset/crop_data/images/train/fracture'
    label_folder = '/Users/huytrq/Workspace/unicas/AIA&ML/Wrist-Fracture-Detection/MLDataset/crop_data/labels/train/fracture'
    output_image_folder = 'crop_aug/images/train/fracture'
    output_label_folder = 'crop_aug/labels/train/fracture'

    generate_augmented_data(image_folder, label_folder, output_image_folder, output_label_folder, num_augmented=3)
