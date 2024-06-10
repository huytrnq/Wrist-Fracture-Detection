# import os

# def filter_yolo_labels(input_dir, output_dir, class_ids_to_filter):
#     """
#     Filters out specific class labels from YOLO format annotation files.

#     Parameters:
#     - input_dir: Directory containing the original YOLO annotation files.
#     - output_dir: Directory to save the filtered annotation files.
#     - class_ids_to_filter: List of class IDs to filter out.

#     Returns:
#     - None
#     """

#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

#     for filename in os.listdir(input_dir):
#         if filename.endswith(".txt"):
#             input_path = os.path.join(input_dir, filename)
#             output_path = os.path.join(output_dir, filename)
            
#             with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
#                 for line in infile:
#                     parts = line.split()
#                     class_id = int(parts[0])
#                     if class_id in class_ids_to_filter:
#                         parts[0] = '0'  # Replace the class ID with 0
#                         outfile.write(' '.join(parts) + '\n')

# if __name__ == "__main__":
#     # Directory containing the original YOLO annotation files
#     input_dir = '/Users/huytrq/Downloads/fracture'
    
#     # Directory to save the filtered annotation files
#     output_dir = '/Users/huytrq/Downloads/fracture1'
    
#     # List of class IDs to filter out
#     class_ids_to_filter = [3]  # Replace with the class IDs you want to filter out
    
#     filter_yolo_labels(input_dir, output_dir, class_ids_to_filter)

import json
import os
import cv2

image_dir = '/Users/huytrq/Workspace/unicas/AIA&ML/Wrist-Fracture-Detection/results/images'

# Load the JSON file
with open('results/DeepLearning/yolov9_e_c_640_val_60epochs_lr001/best_predictions.json') as f:
    data = json.load(f)


# Function to convert bbox to YOLO format
def convert_bbox_to_yolo(bbox, img_width, img_height):
    x_min, y_min, width, height = bbox
    center_x = x_min + width / 2
    center_y = y_min + height / 2
    return [center_x / img_width, center_y / img_height, width / img_width, height / img_height]

# Directory to save the text files
output_dir = 'results/DeepLearning/yolov9_e_c_640_val_60epochs_lr001/txt'
os.makedirs(output_dir, exist_ok=True)

# Convert data to YOLO format and save to text files
for item in data:
    image_id = item['image_id']
    image_path = os.path.join(image_dir, image_id + '.png')
    image = cv2.imread(image_path)
    image_height, image_width = image.shape[:2]
    yolo_bbox = convert_bbox_to_yolo(item['bbox'], image_width, image_height)
    yolo_item = [item['category_id'], item['score']] + yolo_bbox
    yolo_str = ' '.join(map(str, yolo_item))
    
    # Save to a text file named after the image_id
    output_path = os.path.join(output_dir, f"{item['image_id']}.txt")
    with open(output_path, 'a') as f:
        f.write(yolo_str + '\n')

print(f"YOLO formatted data has been saved to text files in {output_dir}")

