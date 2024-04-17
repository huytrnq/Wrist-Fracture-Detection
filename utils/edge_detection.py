import cv2
import numpy as np
import tkinter as tk
from tkinter import Scale
from glob import glob

class Canny:
    def __init__(self, kernel_size):
        self.kernel_size = kernel_size

    def __call__(self, image, threshold1, threshold2):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image
        blurred = cv2.GaussianBlur(gray, (self.kernel_size, self.kernel_size), 0)
        edges = cv2.Canny(blurred, threshold1, threshold2)
        return edges
    
def update_canny():
    threshold1 = threshold1_scale.get()
    threshold2 = threshold2_scale.get()
    edges = conv(img, threshold1, threshold2)
    cv2.imshow('Results', edges)
    if cv2.waitKey(1) == ord('q'):  # Exit if 'q' is pressed
        cv2.destroyAllWindows()
        root.quit()
    else:
        root.after(100, update_canny)  # Call update_canny() every 100 milliseconds

def next_image():
    global img
    image_path = next(image_paths, None)
    if image_path is not None:
        img = cv2.imread(image_path)
    else:
        cv2.destroyAllWindows()
        root.quit()

if __name__ == '__main__':
    path = '/Users/huytrq/Downloads/Compress/Extracted/folder_structure/supervisely/wrist/img/'
    image_paths = iter(glob(path + '*.png'))
    
    conv = Canny(kernel_size=5)
    img = cv2.imread(next(image_paths))

    root = tk.Tk()
    root.title("Canny Thresholds")
    
    threshold1_scale = Scale(root, from_=0, to=255, orient=tk.HORIZONTAL, label="Threshold 1", length=300)
    threshold1_scale.set(30)
    threshold1_scale.pack()

    threshold2_scale = Scale(root, from_=0, to=255, orient=tk.HORIZONTAL, label="Threshold 2", length=300)
    threshold2_scale.set(50)
    threshold2_scale.pack()

    next_button = tk.Button(root, text="Next Image", command=next_image)
    next_button.pack()

    update_canny()  # Start the automatic update

    root.mainloop()
    
    cv2.destroyAllWindows()  # Destroy the windows after the loop completes
