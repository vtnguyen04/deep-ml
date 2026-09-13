import numpy as np

def make_diagonal(x):
	# Your code here
	x = np.asarray(x, dtype=np.float32)
	return np.eye(len(x)) * x