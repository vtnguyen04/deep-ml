import numpy as np

def compute_norm(arr: np.ndarray, norm_type: str) -> float:
    """
    Compute the specified norm of the input array.

    'l1', 'l2' and 'linf' are entrywise norms and accept a 1D or 2D array.
    'frobenius' is a matrix norm and must raise a ValueError if arr is not 2D.

    Args:
        arr: Input numpy array (1D or 2D)
        norm_type: Type of norm ('l1', 'l2', 'linf', or 'frobenius')

    Returns:
        The computed norm as a float
    """
    if arr.ndim not in (1, 2):
        raise ValueError("Input array must be 1D or 2D.")

    norm_type = norm_type.lower()

    if norm_type == 'l1':
        return float(np.sum(np.abs(arr)))

    elif norm_type == 'l2':
        return float(np.sqrt(np.sum(np.square(arr))))

    elif norm_type == 'linf':
        return float(np.max(np.abs(arr)))

    elif norm_type == 'frobenius':

        if arr.ndim != 2:
            raise ValueError("Frobenius norm is only defined for 2D arrays.")
        return float(np.sqrt(np.sum(np.square(arr))))

    else:
        raise ValueError(f"Unsupported norm type: '{norm_type}'")