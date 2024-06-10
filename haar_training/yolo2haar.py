import cv2
import os
import glob

def convert_annotations(txt_path, image_path, output_path):
    with open(txt_path, 'r') as file:
        lines = file.readlines()
    
    lines = [line for line in lines if line[0] == '3']
    # Load the image to get dimensions
    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    # Prepare the annotation string
    annotation = f"{image_path} {len(lines)}"

    for line in lines:
        class_id, x_center, y_center, w, h = map(float, line.split())
        x_center *= width
        y_center *= height
        w *= width
        h *= height
        
        x = int(x_center - w / 2)
        y = int(y_center - h / 2)
        w = int(w)
        h = int(h)
        
        if x < 0:
            x = 0
        if y < 0:
            y = 0
        if x + w > width:
            w = width - x
        if y + h > height:
            h = height - y
        
        annotation += f" {x} {y} {w} {h}"

    with open(output_path, 'a') as file:
        file.write(annotation + '\n')
    return len(lines)

if __name__ == '__main__':
    mode = 'train'
    # dataset_root = '/Users/huytrq/Workspace/unicas/AIA&ML/Wrist-Fracture-Detection/MLDataset/'
    dataset_root = '/Users/huytrq/Workspace/unicas/AIA&ML/Wrist-Fracture-Detection/MLDataset/crop_data/images/train/'
    label_root = '/Users/huytrq/Workspace/unicas/AIA&ML/Wrist-Fracture-Detection/MLDataset/crop_data/labels/train/'
    class_name = 'fracture'
    
    total_count = 0
    for image_path in glob.glob(os.path.join(dataset_root, class_name, '*.png')):
        image_name = os.path.basename(image_path)
        txt_path = os.path.join(label_root, class_name, image_name.replace('.png', '.txt'))
        output_path = 'annotations.txt'

        count_per_file = convert_annotations(txt_path, image_path, output_path)
        total_count += count_per_file
        print(f'Converted {count_per_file} annotations from {image_path}')
    print(f'Total annotations: {total_count}')
    
# 524
# 1819