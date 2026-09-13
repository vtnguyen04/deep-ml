import numpy as np

def pairwise_cosine_similarity(X):

    X = np.asarray(X, dtype=float)

    norms = np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-8)

    X_norm = X / norms
    return X_norm @ X_norm.T