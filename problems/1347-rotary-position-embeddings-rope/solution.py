import torch
import torch.nn as nn


def build_rope_cache(seq_len, head_dim, base=10000.0):
    # TODO: inv_freq = 1 / (base ** (arange(0, head_dim, 2) / head_dim))
    # TODO: theta = outer(arange(seq_len), inv_freq)  -> (seq_len, head_dim // 2)
    # TODO: return (cos(theta), sin(theta))
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))

    t = torch.arange(seq_len, dtype=torch.float32)

    theta = torch.outer(t, inv_freq)

    cos = torch.cos(theta)
    sin = torch.sin(theta)

    return cos, sin


def apply_rope(x, cos, sin):
    # TODO: split x into halves x1, x2 along the last dim
    # TODO: rotated = cat(-x2, x1)
    # TODO: return x * cat(cos, cos) + rotated * cat(sin, sin)
    head_dim = x.shape[-1]

    d_half = head_dim // 2

    cos = cos.to(device=x.device, dtype=x.dtype)
    sin= sin.to(device=x.device, dtype=x.dtype)

    x1 = x[..., :d_half]
    x2 = x[..., d_half:]

    rotated = torch.cat((-x2, x1), dim=-1)

    cos_cat = torch.cat((cos, cos), dim=-1)
    sin_cat = torch.cat((sin, sin), dim=-1)

    return x * cos_cat + rotated * sin_cat

