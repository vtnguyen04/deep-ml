import numpy as np

def recall(y_true, y_pred):
    """
    Calculate the recall metric for binary classification.
    
    Args:
        y_true: Array of true binary labels (0 or 1)
        y_pred: Array of predicted binary labels (0 or 1)
    
    Returns:
        Recall value as a float
    """
    true_positives = np.sum((y_true == 1) & (y_pred == 1))

    actual_positives = np.sum(y_true == 1)

    if actual_positives == 0:
        return 0.0

    return float(true_positives / actual_positives)