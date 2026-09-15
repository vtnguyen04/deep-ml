import numpy as np


def explained_variance_ratio(X: list | np.ndarray) -> list[float]:
 
    X = np.asarray(X, dtype=float)
    n_samples = X.shape[0]

    X_centered = X - np.mean(X, axis=0)

    cov_matrix = (X_centered.T @ X_centered) / (n_samples - 1)
    eigenvalues, _ = np.linalg.eigh(cov_matrix)

    sorted_eigenvalues = np.sort(eigenvalues)[::-1]
    sorted_eigenvalues = np.maximum(sorted_eigenvalues, 0.0)

    total_variance = np.sum(sorted_eigenvalues)
    if total_variance == 0:
        return [0.0] * len(sorted_eigenvalues)

    ratios = sorted_eigenvalues / total_variance
    return ratios