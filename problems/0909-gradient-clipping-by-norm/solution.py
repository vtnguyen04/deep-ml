import torch

def clip_grad_norm(parameters, max_norm: float) -> float:
    # TODO: compute total grad norm, scale in-place if it exceeds max_norm, return original norm
    params_with_grad = [p for p in parameters if p.grad is not None]

    if not params_with_grad:
        return 0.0

    total_norm_sq = sum(p.grad.detach().norm(2).item() ** 2 for p in params_with_grad)
    total_norm = total_norm_sq ** 0.5

    if total_norm > max_norm:
        scale = max_norm / total_norm

        for p in params_with_grad:
            p.grad.mul_(scale)

    return float(total_norm)