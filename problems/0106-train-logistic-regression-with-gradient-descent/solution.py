import numpy as np

def train_logreg(X: np.ndarray, y: np.ndarray, learning_rate: float, iterations: int) -> tuple[list[float], ...]:
    """
    Gradient-descent training algorithm for logistic regression, optimizing parameters with Binary Cross Entropy loss.
    """

	
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    n_samples, n_features = X.shape
	X_b = np.hstack([np.ones((n_samples, 1)), X])

    weights = np.zeros(n_features + 1, dtype=float)

	losses = []
    for _ in range(iterations):
   
        logits = X_b @ weights
        y_pred = 1.0 / (1.0 + np.exp(-np.clip(logits, -500, 500)))

		eps = 1e-15
		y_clipped = np.clip(y_pred, eps, 1 - eps
		)
		loss = -np.sum(y * np.log(y_clipped) + (1 - y) * np.log(1 - y_clipped))
		losses.append(round(float(loss), 4))

        error = y_pred - y
        dw = X_b.T @ error
        weights -= learning_rate * dw

    return np.round(weights, 4).tolist(), losses