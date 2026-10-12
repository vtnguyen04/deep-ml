import torch

def linear_backward(grad_output, x, W):
    # TODO: return (grad_input, grad_W, grad_b) for y = x @ W.T + b
    grad_input = grad_output @ W
    
    # grad_W = grad_output.T @ x
    # Shape: (out_features, N) @ (N, in_features) -> (out_features, in_features)
    grad_W = grad_output.T @ x
    
    # Shape: (N, out_features) -> (out_features,)
    grad_b = grad_output.sum(dim=0)
    
    return grad_input, grad_W, grad_b
