import numpy as np

def random_split(data: np.ndarray, train_frac: float, validation_frac: float, seed: int = 123) -> list:
    """
    Randomly split a dataset into train, validation, and test subsets.
    """
    # Your code here
    n = data.shape[0]
    rng = np.random.default_rng(seed = seed)
    shuffle_idx = rng.permutation(n)
    shuffled_data = data[shuffle_idx]
    train_end = int(n * train_frac)
    val_end = train_end + int(n * validation_frac)
    return [shuffled_data[: train_end], shuffled_data[train_end: val_end], shuffled_data[val_end:]]