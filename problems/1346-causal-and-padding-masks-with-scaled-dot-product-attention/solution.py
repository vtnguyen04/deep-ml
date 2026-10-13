import torch
import torch.nn as nn
import torch.nn.functional as F


def attention(q, k, v, key_padding_mask=None, causal=False):
    # TODO: build a single bool attn_mask broadcastable to (B, H, T_q, T_k)
    #       where True means "may attend"
    # TODO: combine the causal lower-triangular mask and the key padding mask
    # TODO: return F.scaled_dot_product_attention(q, k, v, attn_mask=...)
    batch = q.size(0)

    seq_q = q.size(-2)
    seq_k = k.size(-2)
    
    device = q.device

    attn_mask = None

    if key_padding_mask is not None:

        attn_mask = key_padding_mask.view(batch, 1, 1, seq_k)

    if causal:

        causal_mask = torch.ones((seq_q, seq_k), dtype = torch.bool, device=device).tril()

        if attn_mask is not None:
            attn_mask = attn_mask & causal_mask
        
        else :
            attn_mask = causal_mask

    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask,
        is_causal=False
    )