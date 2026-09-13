import numpy as np

def low_rank_approximation(delta_W: np.ndarray, r: int) -> list:
	"""
	Compute the best rank-r approximation of delta_W via truncated SVD.

	Args:
		delta_W: matrix of shape (m, n)
		r: target rank (1 <= r <= min(m, n))

	Returns:
		The rank-r approximation as a nested Python list of shape (m, n).
	"""
	U, S, Vt = np.linalg.svd(delta_W, full_matrices=False)
    
    W_approx = (U[:, :r] * S[:r]) @ Vt[:r, :]
    
    return W_approx.tolist()