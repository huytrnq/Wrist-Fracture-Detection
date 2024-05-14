
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
