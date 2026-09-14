import torch

def min_max(x: torch.Tensor) -> torch.Tensor:
    """
    Perform Min-Max normalization to scale values to [0, 1].
    
    Args:
        x: A tensor of numerical values
    
    Returns:
        A new tensor with values normalized to [0, 1]
    """
    # Your code here

    x = x.float()
    x_min = torch.min(x)
    x_max = torch.max(x)

    if x_max == x_min:
        return torch.zeros_like(x)

    return (x - x_min) / (x_max - x_min)