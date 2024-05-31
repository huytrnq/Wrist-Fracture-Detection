
### calculate sentitivity and specificity
# Path: utils/metrics.py
def calculate_sensitivity(ground_truths, predictions):
    """
    Calculate the sensitivity (true positive rate) given ground truths and predictions.

    Args:
    ground_truths (list of int): List of ground truth values (0 or 1).
    predictions (list of int): List of prediction values (0 or 1).

    Returns:
    float: Sensitivity (true positive rate).
    """
    true_positives = 0
    false_negatives = 0

    for gt, pred in zip(ground_truths, predictions):
        if gt == 1 and pred == 1:
            true_positives += 1
        elif gt == 1 and pred == 0:
            false_negatives += 1

    sensitivity = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) != 0 else 0

    return sensitivity


import numpy as np

class ObjectDetectionMetrics:
    def __init__(self, iou_threshold=0.5):
        self.iou_threshold = iou_threshold
        self.ground_truths = []
        self.predictions = []
    
    def yolo_to_bbox(self, yolo_bbox, image_shape):
        x_center, y_center, w, h = yolo_bbox
        height, width = image_shape[:2]
        
        x_center *= width
        y_center *= height
        w *= width
        h *= height
        
        x = int(x_center - w / 2)
        y = int(y_center - h / 2)
        w = int(w)
        h = int(h)
        
        return x, y, x + w, y + h

    def add_ground_truth(self, image_id, bbox, type='yolo'):
        if type == 'yolo':
            bbox = self.yolo_to_bbox(bbox, (256, 256))
        self.ground_truths.append((image_id, bbox))

    def add_prediction(self, image_id, bbox):
        self.predictions.append((image_id, bbox))

    def calculate_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        
        inter_width = max(0, xB - xA)
        inter_height = max(0, yB - yA)
        inter_area = inter_width * inter_height
        
        boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        
        iou = inter_area / float(boxA_area + boxB_area - inter_area)
        
        return iou

    def evaluate(self):
        tp = 0
        fp = 0
        fn = 0
        iou_scores = []

        gt_dict = {}
        pred_dict = {}

        for gt in self.ground_truths:
            image_id, bbox = gt
            if image_id not in gt_dict:
                gt_dict[image_id] = []
            gt_dict[image_id].append(bbox)

        for pred in self.predictions:
            image_id, bbox = pred
            if image_id not in pred_dict:
                pred_dict[image_id] = []
            pred_dict[image_id].append(bbox)

        for image_id in gt_dict:
            gt_bboxes = gt_dict[image_id]
            if image_id in pred_dict:
                pred_bboxes = pred_dict[image_id]
                matched_gt = set()
                for pred_bbox in pred_bboxes:
                    best_iou = 0
                    best_gt_idx = -1
                    for gt_idx, gt_bbox in enumerate(gt_bboxes):
                        if gt_idx in matched_gt:
                            continue
                        iou = self.calculate_iou(pred_bbox, gt_bbox)
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = gt_idx
                    if best_iou >= self.iou_threshold:
                        tp += 1
                        iou_scores.append(best_iou)
                        matched_gt.add(best_gt_idx)
                    else:
                        fp += 1
                fn += len(gt_bboxes) - len(matched_gt)
            else:
                fn += len(gt_bboxes)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        miou = np.mean(iou_scores) if iou_scores else 0

        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "mIOU": miou
        }