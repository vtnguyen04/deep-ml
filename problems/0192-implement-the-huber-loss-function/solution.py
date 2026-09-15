import numpy as np
def huber_loss(y_true, y_pred, delta=1.0):
	"""
	Compute the Huber Loss between true and predicted values.

	Args:
		y_true (float | list[float]): Ground truth values
		y_pred (float | list[float]): Predicted values
		delta (float): Transition threshold between MSE and MAE behavior

	Returns:
		float: Average Huber loss
	"""
	# Your code here
	y_true = np.asarray(y_true, dtype=float)
	y_pred = np.asarray(y_pred, dtype=float)

	abs_error = np.abs(y_true - y_pred)

	loss = np.where(
		abs_error <= delta,
		0.5 * (abs_error**2),
		delta * (abs_error - 0.5 * delta),
	)

	return float(np.mean(loss))