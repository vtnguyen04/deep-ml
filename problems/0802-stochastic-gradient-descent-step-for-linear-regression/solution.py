import numpy as np

def sgd_update(X: np.ndarray, y: np.ndarray, weights: np.ndarray, learning_rate: float, n_iter: int) -> list:
    """
    Perform n_iter steps of stochastic gradient descent on a linear regression
    model with MSE loss, cycling through samples in order.

    Returns the final weight vector as a Python list.
    """
    n_sample = X.shape[0]
    for i in range(n_iter):
        idx = i % n_sample

        x_i = X[idx]
        y_i = y[idx]

        y_pred = x_i @ w

        grad = 2.0 * (y_pred - y_i) * x_i

        weights -= learning_rate * grad
       
    return weights

