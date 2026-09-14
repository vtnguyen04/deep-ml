import numpy as np

def precision_recall_at_threshold(y_true, y_scores, threshold):
    """
    Compute precision and recall at a given decision threshold.

    Args:
        y_true: list/array of true binary labels (0 or 1)
        y_scores: list/array of predicted scores in [0, 1]
        threshold: float, classification threshold (predict positive if score >= threshold)

    Returns:
        [precision, recall] as a list of two floats rounded to 4 decimals.
    """
    # Your code here
    tp = 0
    pred_positives = 0
    actual_positives = 0

    for true, score in zip(y_true, y_scores):
        pred = score >= threshold

        if pred:
            pred_positives += 1
            if true == 1:
                tp += 1

        if true == 1:
           actual_positives += 1

    precision = (tp / pred_positives) if pred_positives > 0 else 0.0
    recall = (tp / actual_positives) if actual_positives > 0 else 0.0

    return [round(float(precision), 4), round(float(recall), 4)]
