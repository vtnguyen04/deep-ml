import numpy as np

def mini_batch_gd_step(X: np.ndarray, y: np.ndarray, weights: np.ndarray, bias: float, batch_indices: list, lr: float) -> np.ndarray:
    """
    Perform one mini-batch gradient descent update step for linear regression with MSE loss.
    Returns a 1D array of length D+1: updated weights followed by updated bias.
    """
    theta = np.append(weights, bias)
    
    X_b = X[batch_indices]
    m = len(batch_indices)
    ones = np.ones((m, 1))
    X_b_aug = np.hstack([X_b, ones]) 

    error = X_b_aug @ theta - y[batch_indices]
    grad = (2.0 / m) * (X_b_aug.T @ error)
    
    return theta - lr * grad