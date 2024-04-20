import os
import json
from glob import glob

# Path to the JSON file
root_path = './../dataset/'
ann_folder = 'ann'
img_folder = 'img'
txt_folder = 'txt'

categories = ['fracture', 'normal']

for cat in categories:
    os.makedirs(os.path.join(root_path, img_folder, cat), exist_ok=True)
    os.makedirs(os.path.join(root_path, ann_folder, cat), exist_ok=True)
    os.makedirs(os.path.join(root_path, txt_folder, cat), exist_ok=True)

for file_path in glob(os.path.join(root_path, ann_folder, '*.json')):
    image_name = os.path.basename(file_path).split('.')[0] + '.png'
    json_name = os.path.basename(file_path)
    txt_name = os.path.basename(file_path).split('.')[0] + '.txt'
    
    # Load the JSON content
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    for object in data['objects']:
        try:
            if object['classTitle'] == 'fracture':
                os.rename(os.path.join(root_path, img_folder, image_name), os.path.join(root_path, img_folder, 'fracture', image_name))
                os.rename(os.path.join(root_path, ann_folder, json_name), os.path.join(root_path, ann_folder, 'fracture', json_name))
                # os.rename(os.path.join(root_path, txt_folder, txt_name), os.path.join(root_path, txt_folder, 'fracture', txt_name))
                break
            else:
                os.rename(os.path.join(root_path, img_folder, image_name), os.path.join(root_path, img_folder, 'normal', image_name))
                os.rename(os.path.join(root_path, ann_folder, json_name), os.path.join(root_path, ann_folder, 'normal', json_name))
                # os.rename(os.path.join(root_path, txt_folder, txt_name), os.path.join(root_path, txt_folder, 'normal', txt_name))
        except Exception as e:
            print(e)
