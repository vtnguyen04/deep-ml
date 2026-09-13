import numpy as np

def newton_schulz(M, num_iters: int, a: float, b: float, c: float):
    """
    Apply Newton-Schulz iterations to approximately orthogonalize M.
    Returns the resulting matrix as a nested list of floats.
    """
    # Your code here
    
    M = np.asarray(M, dtype=float)

    frob_norm = np.linalg.norm(M)

    if frob_norm == 0:
        return M
    M /= frob_norm

    for _ in range(num_iters):

        A = M @ M.T
        B = A @ M
        C = A @ B

        M = a * M + b * B + c * C
    return M


