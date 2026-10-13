import torch
import torch.nn as nn
import torch.nn.functional as F

class KVCache:
    def __init__(self):
        # TODO: start empty
        self.k = None
        self.v = None

    def append(self, k, v):
        # TODO: concatenate k, v onto the cache along the sequence dim (dim=2)
        # TODO: return the full (k, v) accumulated so far
        
        if self.k is None:
            self.k = k
            self.v = v

        else:
            self.k = torch.cat([self.k, k], dim=2)
            self.v = torch.cat([self.v, v], dim=2)
        return self.k, self.v

    def __len__(self):
        # TODO: number of cached positions, 0 when empty
        if self.k is None:
            return 0
        return self.k.shape[2]

    def reset(self):
        # TODO: drop everything
        self.k = None
        self.v = None


def attend_with_cache(q, k_new, v_new, cache):
    # TODO: append the new keys/values, then attend q over the full cache
    # TODO: softmax(q @ k^T / sqrt(head_dim)) @ v

    k_full, v_full = cache.append(k_new, v_new)

    head_dim = q.shape[-1]

    scale = head_dim ** 0.5

    score = torch.matmul(q, k_full.transpose(-1, -2)) /scale
    attn_weights = F.softmax(score, dim=-1)

    output = torch.matmul(attn_weights, v_full)

    return output