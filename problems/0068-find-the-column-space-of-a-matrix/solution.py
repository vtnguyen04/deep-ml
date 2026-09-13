import torch

def matrix_image(A: torch.Tensor) -> torch.Tensor:

    A_float = A.to(torch.float64)
    n_cols = A.shape[1]
    
    selected_cols = []
    current_rank = 0
    
    for i in range(n_cols):
        candidate = A_float[:, selected_cols + [i]]
        new_rank = int(torch.linalg.matrix_rank(candidate).item())
        
        if new_rank > current_rank:
            selected_cols.append(i)
            current_rank = new_rank
            
    return A[:, selected_cols]