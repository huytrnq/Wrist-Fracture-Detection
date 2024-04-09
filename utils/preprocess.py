import os
import cv2

from pathlib import Path
from glob import glob

IMG_FORMATS = "bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp", "pfm" 

def resize_with_ratio(img, size):
    """Resize the image with the original aspect ratio

    Args:
        img (np.array): image to be resized
        size (tuple): new size of the image

    Returns:
        np.array: resized image
    """
    h, w = img.shape[:2]
    if h > w:
        h, w = size
        w = int(w * (w / h))
    else:
        w, h = size
        h = int(h * (h / w))
    return cv2.resize(img, (w, h))

class Preprocessor:
    def __init__(self, path, img_size, transforms=None) -> None:
        """Initializes the Preprocessor class

        Args:
            path (str): path to the images
            img_size (tuple): working size of the images
        """
        
        self.transforms = transforms
        
        if isinstance(path, str) and Path(path).suffix == '.txt':
            path = Path(path).read_text().splitlines()
        files = []
        for p in sorted(p) if isinstance(path, (list, tuple)) else [path]:
            p = str(Path(p).resolve())
            if "*" in p:
                files.extend(sorted(glob(p, recursive=True))) #glob
            elif os.path.isdir(p):
                files.extend(sorted(glob(os.path.join(p, "*.*")))) #dir
            elif os.path.isfile(p):
                files.append(p)
            else:
                raise FileNotFoundError(f"File not found: {p}")
            
        images = [image for image in files if image.split('.')[-1].lower() in IMG_FORMATS]
        self.files = images
        self.file_count = len(images)
        
    def __len__(self):
        return self.file_count
    
    def __iter__(self):
        """Initializes the iterator by resetting the count and returning the instance"""
        self.count = 0
        return self
    
    def __next__(self):
        if self.count == self.file_count:
            return StopIteration
        path = self.files[self.count]
        img0 = cv2.imread(path)
        self.count += 1
        assert img0 is not None, f"File not found {path}"
        
        if self.transforms:
            img = self.transforms(img0)
        else:
            img = img0
    
        return path, img0, img
    