import cv2
import numpy as np
import random

# Global variables
img = None
img_displayed = None
regions_displayed = None
inputted_markers = None
is_drawing = False
curve_count = 0
prev = (0, 0)

def magic_wand(event, x, y, flags, param):
    global img, img_displayed, regions_displayed, inputted_markers
    global is_drawing, curve_count, prev

    if event == cv2.EVENT_LBUTTONDOWN:
        is_drawing = True
        curve_count += 1
        prev = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE and is_drawing:
        cv2.line(img_displayed, prev, (x, y), (0, 0, 255), 2)
        cv2.line(inputted_markers, prev, (x, y), curve_count, thickness=2)
        prev = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        is_drawing = False
        watershed_markers = inputted_markers.copy()
        cv2.watershed(img, watershed_markers)

        # Map each region to a random color
        unique_labels = np.unique(watershed_markers)
        label_to_color = {label: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for label in unique_labels}

        # Color the regions
        regions_displayed = np.zeros_like(img)
        for label, color in label_to_color.items():
            regions_displayed[watershed_markers == label] = color

        # Find and draw contours
        contours, _ = cv2.findContours((watershed_markers == -1).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img_displayed, contours, -1, (0, 255, 255), 2)

def setup(image_path):
    global img, img_displayed, regions_displayed, inputted_markers

    img = cv2.imread(image_path)
    if img is None:
        raise IOError("Cannot open image")

    img_displayed = img.copy()
    regions_displayed = np.zeros_like(img)
    inputted_markers = np.zeros(img.shape[:2], dtype=np.int32)

def main(image_path):
    setup(image_path)

    cv2.namedWindow("Magic Wand")
    cv2.namedWindow("Segmented Regions")
    cv2.setMouseCallback("Magic Wand", magic_wand)

    while True:
        cv2.imshow("Magic Wand", img_displayed)
        cv2.imshow("Segmented Regions", regions_displayed)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    IMAGE_PATH = '/Users/huytrq/Downloads/Compress/Extracted/folder_structure/supervisely/wrist/img/0116_0626059847_01_WRI-R1_M014.png'
    main(IMAGE_PATH)
