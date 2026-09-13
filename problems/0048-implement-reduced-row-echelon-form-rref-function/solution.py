import numpy as np

def rref(matrix):
	A = np.asarray(matrix)
    if A.ndim != 2 or A.size == 0:
        return A.copy()

    m, n = A.shape
    M = A.astype(np.float64, copy=True)
    r = 0
	pivot_cols = []
    for c in range(n):
        if r >= m:
            break
        pivot_idx = np.argmax(np.abs(M[r:, c])) + r
        if np.abs(M[pivot_idx, c]) <= 1e-8:
            continue

        if pivot_idx != r:
            M[[r, pivot_idx]] = M[[pivot_idx, r]]

        M[r] /= M[r, c]

        factors = M[:, c].copy()
        factors[r] = 0.0  
        M -= factors[:, None] * M[r:r + 1]

        M[:, c] = 0.0
        M[r, c] = 1.0

        pivot_cols.append(c)
        r += 1

    M[np.abs(M) < 1e-8] = 0.0

    return M