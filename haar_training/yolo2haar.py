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
        
        annotation += f" {x} {y} {w} {h}"

    with open(output_path, 'a') as file:
        file.write(annotation + '\n')

if __name__ == '__main__':
    mode = 'train'
    # dataset_root = '/Users/huytrq/Workspace/unicas/AIA&ML/Wrist-Fracture-Detection/MLDataset/'
    dataset_root = '/Users/huytrq/Workspace/unicas/AIA&ML/Wrist-Fracture-Detection/dataset/img'
    label_root = '/Users/huytrq/Workspace/unicas/AIA&ML/Wrist-Fracture-Detection/MLDataset/txt'
    class_name = 'fracture'
    
    for image_path in glob.glob(os.path.join(dataset_root, class_name, '*.png')):
        image_name = os.path.basename(image_path)
        txt_path = os.path.join(label_root, image_name.replace('.png', '.txt'))
        output_path = os.path.join(dataset_root, 'annotations2.txt')

        convert_annotations(txt_path, image_path, output_path)