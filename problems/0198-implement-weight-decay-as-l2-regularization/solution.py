import torch

def apply_weight_decay(parameters: list[torch.Tensor], gradients: list[torch.Tensor], 
                       lr: float, weight_decay: float, apply_to_all: list[bool]) -> list[torch.Tensor]:
    """
    Apply weight decay (L2 regularization) to parameters.
    
    Args:
        parameters: List of parameter tensors
        gradients: List of gradient tensors
        lr: Learning rate
        weight_decay: Weight decay factor
        apply_to_all: Boolean list indicating which parameter groups get weight decay
    
    Returns:
        Updated parameters
    """
    # Your code here

    with torch.no_grad():
        for param, grad, should_decay in zip(parameters, gradients, apply_to_all):
            if should_decay and weight_decay != 0.0:
                param.mul_(1.0 - lr * weight_decay)

            param.add_(grad, alpha=-lr)

    return parameters