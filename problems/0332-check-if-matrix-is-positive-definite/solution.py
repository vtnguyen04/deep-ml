import numpy as np

def check_positive_definite(matrix: list) -> dict:
    """
    Check if a matrix is positive definite and compute its eigenvalues.
    
    Args:
        matrix: A 2D list representing a square matrix
        
    Returns:
        dict with 'is_positive_definite' (bool) and 'eigenvalues' (list of floats sorted ascending)
    """
    # Your code here
    
    mat = np.asarray(matrix)

    is_symetric = np.allclose(mat, mat.T)

    if is_symetric:
        eigenvalues = np.linalg.eigvalsh(mat)
    else:
        eigenvalues = np.linalg.eigvals(mat)
    
    eigenvalues = np.sort(eigenvalues)

    return {
        'is_positive_definite': bool(is_symetric and np.all(eigenvalues > 0)), 
        'eigenvalues': [round(float(val), 4) for val in eigenvalues]
    }

