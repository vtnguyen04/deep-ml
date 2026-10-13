import torch

def dropout(x: torch.Tensor, p: float, training: bool) -> torch.Tensor:
    # TODO: implement inverted dropout
    if not training or p == 0.0:
        return x

    
    mask = (torch.rand_like(x) > p).to(x.dtype)

    scale = 1.0 / (1.0 - p)


    return x * mask * scale
