import os
import cv2
import argparse

from models.models import WristFractureDetection
from models.kernels import HogDescriptor, CannyDescriptor, AlexNetDescriptor

from utils.dataset import load_yolo_labels, DataLoader
from utils.vis import draw_bboxes
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

def args():
    """Parse the arguments for the inference script

    Returns:
        args: Arguments for the inference script
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--infer_path", type=str, default="MLDataset/images/test/fracture")
    parser.add_argument("--export_results", action="store_true")
    parser.add_argument("--save_path", type=str, default="results")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--window_size", type=int, default=64)
    parser.add_argument("--step_size", type=int, default=32)
    parser.add_argument("--pool_kernel", type=tuple, default=(4, 4))
    parser.add_argument("--mean_hist", type=str, default="./npys/mean_hist.npy")
    parser.add_argument("--heatmap", type=str, default="./npys/heatmap.npy")
    parser.add_argument("--model_path", type=str, default="models/weights/fast_region_proposal_svm.pkl")
    parser.add_argument("--scaler", type=str, default="models/weights/scaler_svm.pkl")
    parser.add_argument("--labels_folder", type=str, default=None)
    return parser.parse_args()

if __name__ == "__main__":
    # Configuration and setup
    args = args()
    infer_path = args.infer_path
    export_results = args.export_results
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)

    # Infer parameters
    window_size = args.window_size
    step_size = args.step_size
    pool_kernel = args.pool_kernel


    # Transforms
    transforms = Compose(
        [
            Pooling(kernel_size=pool_kernel, mode="max"), # Apply max pooling to the image
            UnsharpMask(), # Apply unsharp mask to the image
            HistogramEqualization(intensity_crop=0.1), # Equalize the histogram of the image
            Normalize(outputbitdepth=8), # Normalize the image to 8 bit
            HistogramMatching(mean_hist=args.mean_hist), # Match the mean histogram of the image to the mean histogram of the mean histogram
            ROI_Isolation(dilation=4, dilation_radius=1), # Apply masking from otsu segmentation to the image
        ]
    )
    # DataLoader
    data_loader = DataLoader(infer_path, transforms=transforms, heatmap=args.heatmap)

    # Initialize feature extractors
    hog_descriptor = HogDescriptor() # Apply hog descriptor to the image
    canny_descriptor = CannyDescriptor() # Apply canny descriptor to the image
    alexnet_descriptor = AlexNetDescriptor() # Apply alexnet descriptor to the image
    feature_extractors = [hog_descriptor, alexnet_descriptor, canny_descriptor]

    # # Initialize the model for wrist fracture detection
    wfd = WristFractureDetection(
        model_path=args.model_path, # Load the model from the weights
        feature_extractor=feature_extractors, # Apply the feature extractors to the image
        window_size=window_size, # Sliding windwow size
        step_size=step_size, # Step size
        scaler=args.scaler, # Load the scaler from the weights
    )

    # Load the image from the data loader
    for image_path, img0, img, offset in data_loader:
        print('Predicting on ' + image_path)
        # Predict the bounding boxes
        pred_bboxes = wfd.predict(img, offset)

        # Load corresponding labels for the image if labels_folder is provided
        if args.labels_folder is not None:
            label_path = os.path.join(args.labels_folder, os.path.basename(image_path).replace(".png", ".txt"))
            labels = load_yolo_labels(label_path, img0.shape, [3])
        else:
            labels = []
        # Convert the image to BGR
        img0 = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR)
        # Draw the bounding boxes
        for xyxy in pred_bboxes:
            x1, y1, x2, y2 = xyxy
            cv2.rectangle(img0, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img0, 'prediction', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # Draw the labels
        draw_bboxes(img0, labels, (0, 0, 255))
        
        if export_results:
            # Save the results
            cv2.imwrite(os.path.join(save_path, os.path.basename(image_path)), img0)
            print(f"Image saved to {os.path.join(save_path, os.path.basename(image_path))}")   
            label_name = os.path.basename(image_path).replace(".png", ".txt")
            with open(os.path.join(save_path, label_name), "w") as f:
                for xyxy in pred_bboxes:
                    x1, y1, x2, y2 = xyxy
                    f.write(f"0 {x1} {y1} {x2} {y2}\n")

        if args.show:
            # Show the image
            cv2.imshow("image", img0)
            if cv2.waitKey(0) & 0xFF == ord("q"):
                break
    cv2.destroyAllWindows()
