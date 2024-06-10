import os
import cv2
import numpy as np
from glob import glob

from models.models import WristFractureDetection
from models.kernels import HogDescriptor, CannyDescriptor, AlexNetDescriptor

from utils.dataset import load_yolo_labels, DataLoader
from utils.vis import draw_bboxes
from utils.bboxes import convert_to_yolo_format, save_yolo_labels
from utils.transforms import (
    Compose,
    Pooling,
    Padding,
    HistogramMatching,
    HistogramEqualization,
    Normalize,
    UnsharpMask,
    ROI_Isolation,
)

if __name__ == "__main__":
    # Configuration and setup
    export_results = True
    mode = "test"
    class_name = "fracture"
    save_path = os.path.join("results", mode, class_name)
    os.makedirs(save_path, exist_ok=True)
    infer_path = os.path.join("MLDataset/images", mode, class_name)

    # Infer parameters
    window_size = 64
    step_size = 32
    pool_kernel = (4, 4)
    show = True


    # Transforms
    transforms = Compose(
        [
            Pooling(kernel_size=pool_kernel, mode="max"),
            UnsharpMask(),
            HistogramEqualization(intensity_crop=0.1),
            Normalize(outputbitdepth=8),
            HistogramMatching(mean_hist="./mean_hist.npy"),
            ROI_Isolation(dilation=4, dilation_radius=1),
        ]
    )
    # DataLoader
    data_loader = DataLoader(infer_path, transforms=transforms, heatmap="./heatmap.npy")

    # Initialize feature extractors
    hog_descriptor = HogDescriptor()
    canny_descriptor = CannyDescriptor()
    alexnet_descriptor = AlexNetDescriptor()
    feature_extractors = [hog_descriptor, alexnet_descriptor, canny_descriptor]

    # # Initialize the model for wrist fracture detection
    wfd = WristFractureDetection(
        model_path="models/weights/hog_alex_canny.pkl",
        feature_extractor=feature_extractors,
        window_size=window_size,
        step_size=step_size,
        scaler="models/weights/scaler.pkl",
    )

    # Inference
    for image_path, img0, img, offset in data_loader:
        pred_bboxes = wfd.predict(img, offset)

        label_path = image_path.replace("images", "labels").replace(".png", ".txt")
        labels = load_yolo_labels(label_path, img0.shape, [3])

        img0 = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR)
        for xyxy in pred_bboxes:
            x1, y1, x2, y2 = xyxy
            cv2.rectangle(img0, (x1, y1), (x2, y2), (0, 255, 0), 2)
        draw_bboxes(img0, labels, (0, 0, 255))

        if show:
            cv2.imshow("image", img0)
            if cv2.waitKey(0) & 0xFF == ord("q"):
                break
    cv2.destroyAllWindows()
