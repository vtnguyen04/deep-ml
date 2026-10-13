import torch

def cross_entropy(logits, targets):
    # TODO: numerically stable mean cross-entropy
    log_sum_exp = torch.logsumexp(logits, dim=-1)

    b = logits.shape[0]

    target_logits = logits[torch.arange(b, device=logits.device), targets]

    loss = -target_logits + log_sum_exp

    return loss.mean()