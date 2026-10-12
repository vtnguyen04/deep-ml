import torch
from torch.utils.data import Dataset

class TensorPairDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        # TODO: store X and y on the instance
        self.

    def __len__(self):
        # TODO: return the number of samples
        return X.shape[0]

    def __getitem__(self, idx):
        # TODO: return the (x, y) pair at idx
        
        return X[idx], y[idx]
