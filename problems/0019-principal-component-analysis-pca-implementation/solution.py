import numpy as np

def pca(data: np.ndarray, k: int) -> np.ndarray:
    """
    Perform PCA and return the top k principal components.
    
    Args:
        data: Input array of shape (n_samples, n_features)
        k: Number of principal components to return
    
    Returns:
        Principal components of shape (n_features, k), rounded to 4 decimals.
        Each eigenvector's sign is fixed so its first non-zero element is positive.
    """
    # Your code here
    std = np.std(data, axis=0)
    data_centered = (data - np.mean(data, axis = 0)) / np.where(std == 0, 1, std)
    _, S, Vt = np.linalg.svd(data_centered, full_matrices = True)

    components = Vt[:k].T.copy()

    for i in range(k):
        col = components[:, i]
  
        non_zero_idx = np.where(np.abs(col) > 1e-12)[0]
        if len(non_zero_idx) > 0 and col[non_zero_idx[0]] < 0:
            components[:, i] *= -1

    return np.round(components, 4) + 0.0


