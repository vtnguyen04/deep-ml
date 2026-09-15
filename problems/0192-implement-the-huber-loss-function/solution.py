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
	y_true = torch.tensor(y_true, dtype=float)
	y_pred = torch.tensor(y_pred, dtype=float)

	abs_error = torch.abs(y_true - y_pred)

	loss = torch.where(
		abs_error <= delta,
		0.5 * (abs_error**2),
		delta * (abs_error - 0.5 * delta),
	)

	return float(torch.mean(loss))