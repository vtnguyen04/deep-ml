import numpy as np

def f_score(y_true, y_pred, beta):
	"""
	Calculate F-Score for a binary classification task.

	:param y_true: Numpy array of true labels
	:param y_pred: Numpy array of predicted labels
	:param beta: The weight of precision in the harmonic mean
	:return: F-Score rounded to three decimal places
	"""
	tp = np.sum((y_true == 1) & (y_pred == 1))
	fp = np.sum((y_true == 0) & (y_pred == 1))
	fn = np.sum((y_true == 1) & (y_pred == 0))

	beta_sq = beta**2
	numerator = (1 + beta_sq) * tp
	denominator = (1 + beta_sq) * tp + beta_sq * fn + fp

	if denominator == 0:
		return 0.0

	score = numerator / denominator
	return round(float(score), 3)
