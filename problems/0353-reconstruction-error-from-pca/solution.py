import numpy as np

def pca_reconstruction_error(X: np.ndarray, n_components: int) -> float:
    """
    Compute the mean squared reconstruction error from PCA.
    
    Args:
        X: Data matrix of shape (n_samples, n_features)
        n_components: Number of principal components to keep
        
    Returns:
        The mean squared reconstruction error (float)
    """
    mean = np.mean(X, axis = 0)
    X_centered = X - mean

    covariance_matrix = X_centered.T @ X_centered

    eigenvals, eigenvectors = np.linalg.eig (covariance_matrix)

    indices = np.argsort(eigenvals)[::-1][: n_components]

    w = eigenvectors[:, indices]

    reconstruct_pca = (X_centered @ w) @ w.T + mean

    mse = np.mean((X - reconstruct_pca) ** 2)

    return mse





    