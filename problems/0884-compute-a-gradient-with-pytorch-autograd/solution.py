import torch

def grad_of_quadratic(x_value: float) -> float:
    x_torch = torch.tensor(x_value, dtype=torch.float32, requires_grad=True)

    f_x = x_torch ** 2 + 3 * x_torch + 2
    
    f_x.backward()
    
    return x_torch.grad.item()