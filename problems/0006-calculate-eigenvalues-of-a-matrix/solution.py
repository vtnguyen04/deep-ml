import torch

def calculate_eigenvalues(matrix: torch.Tensor) -> torch.Tensor:
    """
    Compute eigenvalues of a 2x2 matrix using PyTorch.
    Input: 2x2 tensor; Output: 1-D tensor with the two eigenvalues in descending order (highest to lowest).
    """
    # Your implementation here
    eigenvals = torch.linalg.eigvals(matrix).real

    sorted_eigenvals, _ = torch.sort(eigenvals, descending=True)
    
    return sorted_eigenvals
