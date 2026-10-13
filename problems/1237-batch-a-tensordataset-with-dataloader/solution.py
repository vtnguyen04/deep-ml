import torch
from torch.utils.data import TensorDataset, DataLoader

def batch_stats(X, y):
    """Wrap X and y in TensorDataset + DataLoader(batch_size=4, shuffle=False).

    Return (num_batches, first_batch_X_shape_tuple).
    """
    # TODO
    dataset = TensorDataset(X, y)

    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

    num_batches = 0


    first_batch_X_shape_tuple = None
    
    for batch_X, batch_y in dataloader:
        if num_batches == 0:
            first_batch_X_shape_tuple = tuple(batch_X.shape)
        num_batches += 1
        
    return num_batches, first_batch_X_shape_tuple