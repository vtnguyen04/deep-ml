import numpy as np

def gaussian_elimination(A, b):
    """
    Solves the system Ax = b using Gaussian Elimination with partial pivoting.
    
    :param A: Coefficient matrix (shape: n x n)
    :param b: Right-hand side vector (shape: n or n x 1)
    :return: Solution vector x
    """

    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    
    n = A.shape[0]
    b_shape = b.shape
    b = b.flatten()
    
    for k in range(n - 1):
        pivot_row = k + np.argmax(np.abs(A[k:, k]))
        
        if np.isclose(A[pivot_row, k], 0.0):
            raise np.linalg.LinAlgError("Matrix is singular or near-singular.")
            
        if pivot_row != k:
            A[[k, pivot_row]] = A[[pivot_row, k]]
            b[[k, pivot_row]] = b[[pivot_row, k]]
            
        factors = (A[k + 1:, k] / A[k, k])[:, None]
        
        A[k + 1:, k:] -= factors * A[k, k:]
        b[k + 1:] -= factors.squeeze() * b[k]

    if np.isclose(A[n - 1, n - 1], 0.0):
        raise np.linalg.LinAlgError("Matrix is singular.")


    x = np.zeros(n, dtype=float)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i + 1:], x[i + 1:])) / A[i, i]

    return x